from nmtg.trainers.nmt_trainer import NMTTrainer
from nmtg.convert import load_checkpoint
from . import register_trainer

@register_trainer('st_nmt')
class StudentTeacherNMTTrainer(NMTTrainer):
    
    @classmethod
    def add_training_options(cls, parser, argv=None):
        super().add_training_options(parser, argv)

        parser.add_argument('-teacher', type=str, required=True,
                            help='Path to the teacher model/checkpoint')
        parser.add_argument('-teacher_weight', type=float, default=1.0,
                            help='Weight to use for the soft targets loss function')
        parser.add_argument('-temperature', type=float, default=1.0,
                            help='Temperature value for the softmax function')
        
    def __init__(self, args, for_training=True, checkpoint=None):
        super().__init__(args, for_training, checkpoint)
        
        ## Build teacher models
        teacherCheckpoint = load_checkpoint(args.teacher)

        self.teacher = self._build_model(teacherCheckpoint['args'])

        if args.cuda:
            self.teacher.cuda()
        if args.fp16:
            self.teacher.half()

        self.teacher.load_state_dict(teacherCheckpoint['model'])
        self.teacher.eval()
        
    def _forward(self, batch, training=True):
        encoder_input = batch.get('src_indices')
        decoder_input = batch.get('tgt_input')
        targets = batch.get('tgt_output')

        if not self.model.batch_first:
            encoder_input = encoder_input.transpose(0, 1).contiguous()
            decoder_input = decoder_input.transpose(0, 1).contiguous()
            targets = targets.transpose(0, 1).contiguous()

        encoder_mask = encoder_input.ne(self.src_dict.pad())
        decoder_mask = decoder_input.ne(self.tgt_dict.pad())

        T = self.args.temperature
        alpha = self.args.teacher_weight
        outputs, attn_out = self.model(encoder_input, decoder_input, encoder_mask, decoder_mask)
        
        teacher_outputs, teacher_attn_out = self.teacher(encoder_input, decoder_input, encoder_mask, decoder_mask)

        lprobs = self.model.get_normalized_probs(outputs, attn_out, encoder_input,
                                            encoder_mask, decoder_mask, log_probs=True)
        
        soft_lprob = self.model.get_normalized_probs(outputs / T, attn_out, encoder_input,
                                            encoder_mask, decoder_mask, log_probs=True)
        
        soft_target = self.teacher.get_normalized_probs(teacher_outputs / T, teacher_attn_out, encoder_input,
                                    encoder_mask, decoder_mask, log_probs=False)

        if training:
            targets = targets.masked_selected(decoder_mask)
        hard_loss, _ = self.loss(lprobs, targets)
        
        ## Calculate soft/distillation loss
        soft_lprob = soft_lprob.view(-1, soft_lprob.size(-1))
        soft_target = soft_target.view(-1, soft_target.size(-1))

        distillation_loss = -(soft_target * soft_lprob)
        distillation_loss = distillation_loss.sum()

        loss = alpha * distillation_loss + (1 - alpha) * hard_loss
        return loss, loss.item()

