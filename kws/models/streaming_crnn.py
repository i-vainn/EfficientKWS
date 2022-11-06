import torch
import torch.nn.functional as F
from kws.models import CRNN
from kws.augmentations import LogMelspec


# class StreamCRNN(CRNN):
#     '''
#         Streaming version of CRNN.
#     '''

#     def __init__(self, config):
#         super().__init__(config)

#         self.max_window_length = config.max_window_length
#         self.receptive = config.kernel_size[1] - config.stride[1]
#         self.buffer = None
#         self.gru_buffer = None

#     @torch.no_grad()
#     def inference(self, input, hidden=None):
#         self.buffer = torch.cat([self.buffer, input], dim=-1)[:, :, -self.max_window_length:]

#         input = self.buffer[:, :, -self.receptive-input.size(-1):].unsqueeze(dim=1)
#         conv_output = self.conv(input).transpose(-1, -2)
#         gru_output, hidden = self.gru(conv_output, hidden)
#         self.gru_buffer = torch.cat([self.gru_buffer, gru_output], dim=1)
#         contex_vector = self.attention(self.gru_buffer)
#         logits = self.classifier(contex_vector)
#         probs = F.softmax(logits, dim=-1)

#         return probs, hidden

#     def set_buffer(self, batch_size):
#         self.buffer = torch.zeros(
#             size=(batch_size, self.config.n_mels, self.receptive),
#             device=self.config.device
#         )
#         self.gru_buffer = torch.Tensor([], device=self.config.device)


class StreamCRNN(CRNN):
    '''
        Streaming version of CRNN.
    '''

    def __init__(self, config):
        super().__init__(config)

        self.mel_processor = LogMelspec(False, config)
        self.raw_cache = torch.Tensor([])
        self.max_window_length = config.max_window_length
        self.streaming_step_size = config.streaming_step_size
        self.prev_hidden = None
        self.prev_prob = torch.zeros(1, 2)
        self.processed_steps = 0

    def process(self, raw_audio: torch.Tensor):
        self.raw_cache = torch.cat([self.raw_cache, raw_audio])[-self.max_window_length:]
        self.processed_steps += len(raw_audio)

        if self.processed_steps >= self.streaming_step_size:        
            self.processed_steps = 0
            melspec = self.mel_processor(raw_audio).unsqueeze(0)
            logits, self.prev_hidden = super().forward(melspec, self.prev_hidden)
            prob = F.softmax(logits, dim=-1)
            self.prev_prob = prob

        return self.prev_prob
