module MlxLm
  module Tuner
    class TrainingArgs
      DEFAULTS = {
        batch_size: 4,
        iters: 100,
        val_batches: 25,
        steps_per_report: 10,
        steps_per_eval: 200,
        steps_per_save: 100,
        max_seq_length: 2048,
        adapter_file: "adapters.safetensors",
        grad_checkpoint: false,
        grad_accumulation_steps: 1,
      }.freeze

      ATTRS = DEFAULTS.keys.freeze
      attr_accessor(*ATTRS)

      def initialize(**kwargs)
        unknown = kwargs.keys - ATTRS
        unless unknown.empty?
          raise ArgumentError, "unknown TrainingArgs keys: #{unknown.join(', ')}"
        end

        ATTRS.each do |attr|
          public_send("#{attr}=", kwargs.fetch(attr, DEFAULTS[attr]))
        end
      end

      def to_h
        ATTRS.each_with_object({}) do |attr, out|
          out[attr] = public_send(attr)
        end
      end
    end
  end
end
