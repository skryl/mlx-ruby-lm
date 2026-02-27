module MlxLm
  module Tuner
    class TrainingCallback
      def on_train_loss_report(_train_info)
      end

      def on_val_loss_report(_val_info)
      end
    end

    class WandBCallback < TrainingCallback
      attr_reader :wrapped_callback

      def initialize(project_name:, log_dir:, config:, wrapped_callback: nil, client: nil)
        @wrapped_callback = wrapped_callback
        @client = client || resolve_client("Wandb")
        raise LoadError, "wandb client is unavailable" if @client.nil?

        @client.init(
          project: project_name,
          name: File.basename(log_dir.to_s),
          dir: log_dir,
          config: config
        )
      end

      def on_train_loss_report(train_info)
        @client.log(to_serializable(train_info), step: train_info[:iteration] || train_info["iteration"])
        wrapped_callback&.on_train_loss_report(train_info)
      end

      def on_val_loss_report(val_info)
        @client.log(to_serializable(val_info), step: val_info[:iteration] || val_info["iteration"])
        wrapped_callback&.on_val_loss_report(val_info)
      end

      private

      def resolve_client(const_name)
        Object.const_get(const_name)
      rescue NameError
        nil
      end

      def to_serializable(data)
        data.to_h.each_with_object({}) do |(k, v), out|
          out[k] = v.respond_to?(:tolist) ? v.tolist : v
        end
      end
    end

    class SwanLabCallback < TrainingCallback
      attr_reader :wrapped_callback

      def initialize(project_name:, log_dir:, config:, wrapped_callback: nil, client: nil)
        @wrapped_callback = wrapped_callback
        @client = client || resolve_client("Swanlab")
        raise LoadError, "swanlab client is unavailable" if @client.nil?

        @client.init(
          project: project_name,
          experiment_name: File.basename(log_dir.to_s),
          logdir: File.join(log_dir.to_s, "swanlog"),
          config: config
        )
      end

      def on_train_loss_report(train_info)
        @client.log(to_serializable(train_info), step: train_info[:iteration] || train_info["iteration"])
        wrapped_callback&.on_train_loss_report(train_info)
      end

      def on_val_loss_report(val_info)
        @client.log(to_serializable(val_info), step: val_info[:iteration] || val_info["iteration"])
        wrapped_callback&.on_val_loss_report(val_info)
      end

      private

      def resolve_client(const_name)
        Object.const_get(const_name)
      rescue NameError
        nil
      end

      def to_serializable(data)
        data.to_h.each_with_object({}) do |(k, v), out|
          out[k] = v.respond_to?(:tolist) ? v.tolist : v
        end
      end
    end

    SUPPORT_CALLBACK = {
      "wandb" => WandBCallback,
      "swanlab" => SwanLabCallback,
    }.freeze

    module_function

    def get_reporting_callbacks(report_to: nil, project_name: nil, log_dir: nil, config: nil, clients: {})
      return nil if report_to.nil? || report_to.to_s.strip.empty?

      callback_chain = nil
      report_to.to_s.split(",").map(&:strip).map(&:downcase).reject(&:empty?).each do |name|
        klass = SUPPORT_CALLBACK.fetch(name) do
          raise ArgumentError, "#{name} callback doesn't exist choose from #{SUPPORT_CALLBACK.keys.join(', ')}"
        end
        callback_chain = klass.new(
          project_name: project_name,
          log_dir: log_dir,
          config: config,
          wrapped_callback: callback_chain,
          client: clients[name]
        )
      end
      callback_chain
    end
  end
end
