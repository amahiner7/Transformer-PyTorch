import torch.nn as nn
from tensorboardX import SummaryWriter
from torchinfo import summary as summary_
import time
import math

from model.ver2.layers.Encoder import Encoder
from model.ver2.layers.Decoder import Decoder
from config.file_path import *
from config.hyper_parameters import *
from utils.common import *
from model.custom_scheduler.CosineAnnealingWarmRestarts import CosineAnnealingWarmUpRestarts


class Transformer(nn.Module):
    def __init__(self, d_input, d_output, d_embed, d_model, d_ff, num_heads, num_layers,
                 dropout_prob, source_pad_index, target_pad_index, seq_len, device=None, name='Transformer'):
        super().__init__()
        self.name = name
        self.criterion = None
        self.optimizer = None
        self.lr_scheduler = None

        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.encoder = Encoder(d_input=d_input,
                               d_embed=d_embed,
                               d_model=d_model,
                               d_ff=d_ff,
                               num_layers=num_layers,
                               num_heads=num_heads,
                               seq_len=seq_len,
                               dropout_prob=dropout_prob,
                               device=self.device)

        self.decoder = Decoder(d_output=d_output,
                               d_embed=d_embed,
                               d_model=d_model,
                               d_ff=d_ff,
                               num_layers=num_layers,
                               num_heads=num_heads,
                               seq_len=seq_len,
                               dropout_prob=dropout_prob,
                               device=self.device)

        self.source_pad_index = source_pad_index
        self.target_pad_index = target_pad_index

        self.to(self.device)
        print(f"{self.name} : {self.device} is available.")

        def _initialize_weights(m):
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                nn.init.kaiming_uniform_(tensor=m.weight.data, nonlinearity='relu')

        self.apply(_initialize_weights)

    def _count_parameters(self):
        return sum(params.numel() for params in self.parameters() if params.requires_grad)

    def _check_compile(self):
        if self.criterion is None or self.optimizer is None:
            self.compile()

    def _get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def _set_summary_writer(self, tensorboard_dir):
        self.loss_writer = SummaryWriter(tensorboard_dir)
        self.lr_writer = SummaryWriter(os.path.join(tensorboard_dir, "history"))

    def compile(self, criterion=None, optimizer=None):
        if criterion is None:
            criterion = nn.CrossEntropyLoss(ignore_index=self.target_pad_index)
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=0.0)

        self.criterion = criterion
        self.optimizer = optimizer

        self.lr_scheduler = CosineAnnealingWarmUpRestarts(self.optimizer,
                                                          T_0=10,
                                                          T_mult=1,
                                                          eta_max=1e-3,
                                                          T_up=5,
                                                          gamma=0.5)

    # 소스 문장의 <pad> 토큰에 대하여 마스크(mask) 값을 0으로 설정
    def make_source_mask(self, source):
        """
        source shape: (batch_size, source_length)
        """

        # source_mask shape: (batch_size, 1, 1, source_length)
        source_mask = (source != self.source_pad_index).unsqueeze(1).unsqueeze(2)

        return source_mask

    # 타겟 문장에서 각 단어는 다음 단어가 무엇인지 알 수 없도록(이전 단어만 보도록) 만들기 위해 마스크를 사용
    def make_target_mask(self, target):
        """
        target shape: (batch_size, target_length)
        """

        # target_pad_mask shape: (batch_size, 1, 1, target_length)
        target_pad_mask = (target != self.target_pad_index).unsqueeze(1).unsqueeze(2)

        target_length = target.shape[1]

        # target_sub_mask shape: (target_length, target_length)
        target_sub_mask = torch.tril(torch.ones((target_length, target_length), device=self.device)).bool()

        # target_mask shape: (batch_size, 1, target_length, target_length)
        target_mask = target_pad_mask & target_sub_mask

        return target_mask

    def forward(self, source, target):
        """
        source shape: (batch_size, source_len)
        target shape: (batch_size, target_len)
        """

        # source_mask shape: (batch_size, 1, 1, source_len)
        # target_mask shape: (batch_size, 1, target_len, target_len)
        source_mask = self.make_source_mask(source=source)
        target_mask = self.make_target_mask(target=target)

        # encoder_output shape: (batch_size, source_len, model_dim)
        encoder_output = self.encoder(source=source, mask=source_mask)

        # output shape: (batch_size, target_len, output_dim)
        # attention shape: (batch_size, num_heads, target_len, src_len)
        output, attention = self.decoder(decoder_source=target, decoder_mask=target_mask,
                                         encoder_source=encoder_output, encoder_mask=source_mask)

        return output, attention

    def train_on_batch(self, data_loader, log_interval=1):
        loss_list = []
        complete_batch_size = 0

        self._check_compile()
        self.train()  # Train mode

        for batch_index, batch in enumerate(data_loader):
            source = batch.src
            target = batch.trg

            # Initialize gradient
            self.optimizer.zero_grad()

            # Forward propagation
            output, _ = self.forward(source=source, target=target[:, :-1])
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            target = target[:, 1:].contiguous().view(-1)

            # Get loss
            loss = self.criterion(output, target)

            # Back-propagation
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), HyperParameter.CLIP)

            # Update weight
            self.optimizer.step()

            loss_list.append(loss.item())

            complete_batch_size += len(source)
            if (batch_index % log_interval == 0 or (batch_index + 1) == len(data_loader)) and batch_index is not 0:
                print(" BATCH: [{}/{}({:.0f}%)] | TRAIN LOSS: {:.4f}".format(
                    complete_batch_size,
                    len(data_loader.dataset),
                    100.0 * (batch_index + 1) / len(data_loader),
                    loss.item()))

        return loss_list

    def evaluate(self, data_loader):
        loss_list = []
        self.eval()  # Evaluate mode

        with torch.no_grad():
            # 전체 평가 데이터를 확인하며
            for batch_index, batch in enumerate(data_loader):
                source = batch.src
                target = batch.trg

                output, _ = self.forward(source=source, target=target[:, :-1])
                output_dim = output.shape[-1]
                output = output.contiguous().view(-1, output_dim)
                target = target[:, 1:].contiguous().view(-1)
                loss = self.criterion(output, target)

                loss_list.append(loss.item())

        return loss_list

    def train_on_epoch(self, train_data_loader, valid_data_loader, epochs,
                       tensorboard_dir=FilePath.TENSORBOARD_LOG_DIR, log_interval=1):
        best_val_loss = math.inf
        loss_history = []
        val_loss_history = []
        learning_rate_history = []

        self._check_compile()
        self._set_summary_writer(tensorboard_dir)

        for epoch in range(epochs):
            print("=============== TRAINING EPOCHS {} / {} ===============".format(epoch + 1, epochs))
            train_start_time = time.time()

            train_loss_list = self.train_on_batch(data_loader=train_data_loader, log_interval=log_interval)
            val_loss_list = self.evaluate(data_loader=valid_data_loader)

            # Learning rate 업데이트
            self.lr_scheduler.step()

            learning_rate = self._get_lr()

            train_loss = np.mean(train_loss_list)
            val_loss = np.mean(val_loss_list)
            if val_loss < best_val_loss:
                model_file_path = self.save(model_file_dir=FilePath.MODEL_FILE_DIR,
                                            model_file_name=FilePath.MODEL_FILE_NAME,
                                            epoch=epoch,
                                            val_loss=val_loss)
                print("val_loss improved from {:.5f} to {:.5f}, saving model to ".format(
                    best_val_loss, val_loss) + model_file_path)
                best_val_loss = val_loss
            else:
                print("val_loss did not improve from {:.5f}".format(best_val_loss))

            print("TRAIN LOSS: {:.4f}, PPL: {:.4f} | VALID LOSS: {:.4f}, PPL: {:.4f} | LEARNING RATE: {} | "
                  "ELAPSED TIME: {}\n".
                  format(train_loss,
                         math.exp(train_loss),
                         val_loss,
                         math.exp(val_loss),
                         learning_rate,
                         format_time(time.time() - train_start_time)))

            loss_history.append(train_loss)
            val_loss_history.append(val_loss)
            learning_rate_history.append(learning_rate)
            self.loss_writer.add_scalars('history',
                                         {'train': train_loss, 'validation': val_loss},
                                         epoch)
            self.lr_writer.add_scalar('learning_rate/Learning rate', learning_rate, epoch)

        self.loss_writer.close()
        self.lr_writer.close()

        history = {'loss': loss_history, 'val_loss': val_loss_history, 'learning_rate': learning_rate_history}

        return history

    def save(self, model_file_dir, model_file_name, epoch, val_loss):
        if not os.path.exists(model_file_dir):
            os.mkdir(model_file_dir)
            print("Directory: {} is created.".format(model_file_dir))

        model_file_name = model_file_name.format(epoch + 1, val_loss)
        model_file_path = os.path.join(model_file_dir, model_file_name)
        torch.save({'net': self.state_dict(), 'optim': self.optimizer.state_dict()}, model_file_path)

        return model_file_path

    def load(self, model_file_path):
        dict_model = torch.load(model_file_path)
        self._check_compile()
        self.load_state_dict(dict_model['net'])
        self.optimizer.load_state_dict(dict_model['optim'])

    def summary(self, sample_source, sample_target):
        summary_(model=self, input_data=[sample_source, sample_target])
