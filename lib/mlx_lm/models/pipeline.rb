module MlxLm
  module Models
    module PipelineMixin
      attr_accessor :pipeline_rank, :pipeline_size, :start_idx, :end_idx

      def initialize(*args, **kwargs)
        super(*args, **kwargs)
        @pipeline_rank = 0
        @pipeline_size = 1
        @start_idx = 0
        @end_idx = nil
      end

      def pipeline_layers
        layers[@start_idx...@end_idx]
      end

      def pipeline(group)
        # Split layers in reverse so rank=0 gets the last layers and
        # rank=pipeline_size-1 gets the first.
        @pipeline_rank = group.rank
        @pipeline_size = group.size
        layers_per_rank = layers.length / @pipeline_size
        extra = layers.length - (layers_per_rank * @pipeline_size)
        layers_per_rank += 1 if @pipeline_rank < extra

        @start_idx = (@pipeline_size - @pipeline_rank - 1) * layers_per_rank
        @end_idx = @start_idx + layers_per_rank

        self.layers = layers[0...@end_idx]
        # Keep layer numbering stable for checkpoint loading.
        self.layers[0...@start_idx] = Array.new(@start_idx, nil)
        self
      end
    end
  end
end
