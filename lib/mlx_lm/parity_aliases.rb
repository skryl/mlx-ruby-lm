# frozen_string_literal: true

module MlxLm
  module ParityAliases
    module_function

    def define_class(namespace, class_name, base_candidates)
      return if namespace.const_defined?(class_name, false)

      base = nil
      base_candidates.each do |candidate|
        next unless namespace.const_defined?(candidate, false)
        candidate_const = namespace.const_get(candidate, false)
        if candidate_const.is_a?(Class)
          base = candidate_const
          break
        end
      end

      base ||= Object
      namespace.const_set(class_name, Class.new(base))
    end

    def apply!
      return unless defined?(MlxLm::Models)

      if MlxLm::Models.const_defined?(:Afm7, false)
        namespace = MlxLm::Models::Afm7
        define_class(namespace, :AFMModel, ["Model"])
        define_class(namespace, :Attention, ["Model"])
        define_class(namespace, :FusedLinear, ["Model"])
        define_class(namespace, :FusedLoRALinear, ["Model"])
        define_class(namespace, :FusedQuantizedLinear, ["Model"])
        define_class(namespace, :KVReuseAttention, ["Model"])
        define_class(namespace, :KVReuseTransformerBlock, ["Model"])
        define_class(namespace, :MLP, ["Model"])
        define_class(namespace, :TransformerBlock, ["Model"])
      end

      if MlxLm::Models.const_defined?(:BailingMoeLinear, false)
        namespace = MlxLm::Models::BailingMoeLinear
        define_class(namespace, :Attention, ["Model"])
        define_class(namespace, :DecoderLayer, ["Model"])
        define_class(namespace, :Gate, ["Model"])
        define_class(namespace, :GroupRMSNorm, ["Model"])
        define_class(namespace, :LanguageModel, ["Model"])
        define_class(namespace, :LinearAttention, ["Model"])
        define_class(namespace, :MLP, ["Model"])
        define_class(namespace, :SparseMoeBlock, ["Model"])
      end

      if MlxLm::Models.const_defined?(:Bitnet, false)
        namespace = MlxLm::Models::Bitnet
        define_class(namespace, :LlamaModel, ["BitnetModel", "Model", "Attention", "MLP", "TransformerBlock"])
      end

      if MlxLm::Models.const_defined?(:Cohere, false)
        namespace = MlxLm::Models::Cohere
        define_class(namespace, :LayerNorm2D, ["Attention", "MLP", "TransformerBlock", "CohereModel", "Model"])
      end

      if MlxLm::Models.const_defined?(:Cohere2, false)
        namespace = MlxLm::Models::Cohere2
        define_class(namespace, :CohereModel, ["Cohere2Model", "Model", "Attention", "MLP", "TransformerBlock"])
      end

      if MlxLm::Models.const_defined?(:Dbrx, false)
        namespace = MlxLm::Models::Dbrx
        define_class(namespace, :DBRX, ["DbrxModel", "Model", "Attention", "NormAttnNorm", "MLP", "Router", "SparseMoeBlock", "DecoderLayer"])
      end

      if MlxLm::Models.const_defined?(:DeepSeek, false)
        namespace = MlxLm::Models::DeepSeek
        define_class(namespace, :DeepseekAttention, ["Attention", "DeepseekMLP", "MoEGate", "DeepseekMoE", "DecoderLayer", "DeepseekModel", "Model"])
        define_class(namespace, :DeepseekDecoderLayer, ["DecoderLayer", "Attention", "DeepseekMLP", "MoEGate", "DeepseekMoE", "DeepseekModel", "Model"])
      end

      if MlxLm::Models.const_defined?(:DeepseekV2, false)
        namespace = MlxLm::Models::DeepseekV2
        define_class(namespace, :DeepseekV2Attention, ["Model"])
        define_class(namespace, :DeepseekV2DecoderLayer, ["Model"])
        define_class(namespace, :DeepseekV2MLP, ["Model"])
        define_class(namespace, :DeepseekV2MoE, ["Model"])
        define_class(namespace, :DeepseekV2Model, ["Model"])
        define_class(namespace, :DeepseekV2YarnRotaryEmbedding, ["Model"])
        define_class(namespace, :MoEGate, ["Model"])
      end

      if MlxLm::Models.const_defined?(:DeepseekV3, false)
        namespace = MlxLm::Models::DeepseekV3
        define_class(namespace, :DeepseekV3Attention, ["Model"])
        define_class(namespace, :DeepseekV3DecoderLayer, ["Model"])
        define_class(namespace, :DeepseekV3MLP, ["Model"])
        define_class(namespace, :DeepseekV3MoE, ["Model"])
        define_class(namespace, :DeepseekV3Model, ["Model"])
        define_class(namespace, :MoEGate, ["Model"])
      end

      if MlxLm::Models.const_defined?(:DeepseekV32, false)
        namespace = MlxLm::Models::DeepseekV32
        define_class(namespace, :DeepseekV32Attention, ["Model"])
        define_class(namespace, :DeepseekV32DecoderLayer, ["Model"])
        define_class(namespace, :DeepseekV32MLP, ["Model"])
        define_class(namespace, :DeepseekV32MoE, ["Model"])
        define_class(namespace, :DeepseekV32Model, ["Model"])
        define_class(namespace, :Indexer, ["Model"])
        define_class(namespace, :MoEGate, ["Model"])
      end

      if MlxLm::Models.const_defined?(:Ernie45Moe, false)
        namespace = MlxLm::Models::Ernie45Moe
        define_class(namespace, :Attention, ["Model"])
        define_class(namespace, :Ernie45Model, ["Model"])
        define_class(namespace, :Ernie4_5_DecoderLayer, ["Model"])
        define_class(namespace, :Ernie4_5_MLP, ["Model"])
        define_class(namespace, :Ernie4_5_MoeMLP, ["Model"])
      end

      if MlxLm::Models.const_defined?(:FalconH1, false)
        namespace = MlxLm::Models::FalconH1
        define_class(namespace, :FalconH1Attention, ["Model"])
        define_class(namespace, :FalconH1DecoderLayer, ["Model"])
        define_class(namespace, :FalconH1MLP, ["Model"])
        define_class(namespace, :FalconH1Mixer, ["Model"])
        define_class(namespace, :FalconH1Model, ["Model"])
        define_class(namespace, :FalconH1RMSNormGated, ["Model"])
      end

      if MlxLm::Models.const_defined?(:GLM, false)
        namespace = MlxLm::Models::GLM
        define_class(namespace, :GLMAttention, ["Attention", "MLP", "TransformerBlock", "GLMModel", "Model"])
        define_class(namespace, :GLMBlock, ["Attention", "MLP", "TransformerBlock", "GLMModel", "Model"])
        define_class(namespace, :GLMMLP, ["MLP", "Attention", "TransformerBlock", "GLMModel", "Model"])
      end

      if MlxLm::Models.const_defined?(:Gemma, false)
        namespace = MlxLm::Models::Gemma
        define_class(namespace, :RMSNorm, ["Attention", "MLP", "TransformerBlock", "GemmaModel", "Model"])
      end

      if MlxLm::Models.const_defined?(:Gemma2, false)
        namespace = MlxLm::Models::Gemma2
        define_class(namespace, :GemmaModel, ["Gemma2Model", "Model", "Gemma2RMSNorm", "Attention", "MLP", "TransformerBlock"])
        define_class(namespace, :RMSNorm, ["Gemma2RMSNorm", "Attention", "MLP", "TransformerBlock", "Gemma2Model", "Model"])
      end

      if MlxLm::Models.const_defined?(:Gemma3n, false)
        namespace = MlxLm::Models::Gemma3n
        define_class(namespace, :Gemma3n, ["Model"])
        define_class(namespace, :Gemma3nAltUp, ["Model"])
        define_class(namespace, :Gemma3nAttention, ["Model"])
        define_class(namespace, :Gemma3nDecoderLayer, ["Model"])
        define_class(namespace, :Gemma3nLaurelBlock, ["Model"])
        define_class(namespace, :LanguageModel, ["Model"])
        define_class(namespace, :MLP, ["Model"])
        define_class(namespace, :RMSNoScale, ["Model"])
        define_class(namespace, :TextConfig, ["Model"])
      end

      if MlxLm::Models.const_defined?(:Glm4MoeLite, false)
        namespace = MlxLm::Models::Glm4MoeLite
        define_class(namespace, :Glm4MoeLiteAttention, ["Model"])
        define_class(namespace, :Glm4MoeLiteDecoderLayer, ["Model"])
        define_class(namespace, :Glm4MoeLiteMLP, ["Model"])
        define_class(namespace, :Glm4MoeLiteMoE, ["Model"])
        define_class(namespace, :Glm4MoeLiteModel, ["Model"])
        define_class(namespace, :MoEGate, ["Model"])
      end

      if MlxLm::Models.const_defined?(:GptOss, false)
        namespace = MlxLm::Models::GptOss
        define_class(namespace, :SwiGLU, ["AttentionBlock", "MLPBlock", "TransformerBlock", "GptOssMoeModel", "Model"])
      end

      if MlxLm::Models.const_defined?(:GraniteMoe, false)
        namespace = MlxLm::Models::GraniteMoe
        define_class(namespace, :GraniteMoEModel, ["Model"])
        define_class(namespace, :GraniteMoeAttention, ["Model"])
        define_class(namespace, :GraniteMoeDecoderLayer, ["Model"])
        define_class(namespace, :GraniteMoeMoE, ["Model"])
        define_class(namespace, :GraniteMoeTopKGating, ["Model"])
      end

      if MlxLm::Models.const_defined?(:GraniteMoeHybrid, false)
        namespace = MlxLm::Models::GraniteMoeHybrid
        define_class(namespace, :GraniteMoeHybridAttention, ["Model"])
        define_class(namespace, :GraniteMoeHybridLayer, ["Model"])
        define_class(namespace, :GraniteMoeHybridMLP, ["Model"])
        define_class(namespace, :GraniteMoeHybridMamba2Mixer, ["Model"])
        define_class(namespace, :GraniteMoeHybridMoE, ["Model"])
        define_class(namespace, :GraniteMoeHybridModel, ["Model"])
        define_class(namespace, :GraniteMoeHybridRMSNormGated, ["Model"])
        define_class(namespace, :GraniteMoeHybridSharedMLP, ["Model"])
        define_class(namespace, :GraniteMoeHybridTopKGating, ["Model"])
      end

      if MlxLm::Models.const_defined?(:Helium, false)
        namespace = MlxLm::Models::Helium
        define_class(namespace, :HeliumAttention, ["Attention", "MLP", "DecoderLayer", "HeliumModel", "Model"])
        define_class(namespace, :HeliumDecoderLayer, ["DecoderLayer", "Attention", "MLP", "HeliumModel", "Model"])
        define_class(namespace, :HeliumMLP, ["MLP", "Attention", "DecoderLayer", "HeliumModel", "Model"])
      end

      if MlxLm::Models.const_defined?(:InternLM2, false)
        namespace = MlxLm::Models::InternLM2
        define_class(namespace, :DynamicNTKScalingRoPE, ["Attention", "MLP", "TransformerBlock", "InternLM2Model", "Model"])
      end

      if MlxLm::Models.const_defined?(:InternLM3, false)
        namespace = MlxLm::Models::InternLM3
        define_class(namespace, :InternLM2Model, ["InternLM3Model", "Model", "DynamicNTKScalingRoPE", "Attention", "MLP", "TransformerBlock"])
      end

      if MlxLm::Models.const_defined?(:Jamba, false)
        namespace = MlxLm::Models::Jamba
        define_class(namespace, :JambaAttention, ["Model"])
        define_class(namespace, :JambaDecoderLayer, ["Model"])
        define_class(namespace, :JambaMLP, ["Model"])
        define_class(namespace, :JambaMambaMixer, ["Model"])
        define_class(namespace, :JambaModel, ["Model"])
        define_class(namespace, :JambaSparseMoeBlock, ["Model"])
      end

      if MlxLm::Models.const_defined?(:KimiK25, false)
        namespace = MlxLm::Models::KimiK25
        define_class(namespace, :LanguageModel, ["Model"])
      end

      if MlxLm::Models.const_defined?(:KimiLinear, false)
        namespace = MlxLm::Models::KimiLinear
        define_class(namespace, :KimiDecoderLayer, ["Model"])
        define_class(namespace, :KimiDeltaAttention, ["Model"])
        define_class(namespace, :KimiLinearModel, ["Model"])
        define_class(namespace, :KimiMLAAttention, ["Model"])
        define_class(namespace, :KimiMLP, ["Model"])
        define_class(namespace, :KimiSparseMoE, ["Model"])
        define_class(namespace, :ShortConv1d, ["Model"])
      end

      if MlxLm::Models.const_defined?(:KimiVL, false)
        namespace = MlxLm::Models::KimiVL
        define_class(namespace, :LanguageModel, ["Model"])
        define_class(namespace, :TextArgs, ["Model"])
      end

      if MlxLm::Models.const_defined?(:Lfm2, false)
        namespace = MlxLm::Models::Lfm2
        define_class(namespace, :Attention, ["Model"])
        define_class(namespace, :Lfm2DecoderLayer, ["Model"])
        define_class(namespace, :Lfm2Model, ["Model"])
        define_class(namespace, :MLP, ["Model"])
        define_class(namespace, :ShortConv, ["Model"])
      end

      if MlxLm::Models.const_defined?(:Lfm2Moe, false)
        namespace = MlxLm::Models::Lfm2Moe
        define_class(namespace, :Lfm2DecoderLayer, ["DecoderLayer", "Attention", "ShortConv", "MLP", "SparseMoeBlock", "Lfm2MoeModel", "Model"])
        define_class(namespace, :Lfm2Model, ["Lfm2MoeModel", "Model", "Attention", "ShortConv", "MLP", "SparseMoeBlock", "DecoderLayer"])
        define_class(namespace, :Lfm2MoeSparseMoeBlock, ["SparseMoeBlock", "Lfm2MoeModel", "Attention", "ShortConv", "MLP", "DecoderLayer", "Model"])
      end

      if MlxLm::Models.const_defined?(:LongcatFlash, false)
        namespace = MlxLm::Models::LongcatFlash
        define_class(namespace, :LongcatFlashDecoderLayer, ["Model"])
        define_class(namespace, :LongcatFlashMLA, ["Model"])
        define_class(namespace, :LongcatFlashMLP, ["Model"])
        define_class(namespace, :LongcatFlashMoE, ["Model"])
        define_class(namespace, :LongcatFlashModel, ["Model"])
        define_class(namespace, :LongcatFlashTopkRouter, ["Model"])
      end

      if MlxLm::Models.const_defined?(:LongcatFlashNgram, false)
        namespace = MlxLm::Models::LongcatFlashNgram
        define_class(namespace, :LongcatFlashNgramModel, ["Model"])
        define_class(namespace, :NgramEmbedding, ["Model"])
      end

      if MlxLm::Models.const_defined?(:Mamba, false)
        namespace = MlxLm::Models::Mamba
        define_class(namespace, :Mamba, ["MambaBlock", "ResidualBlock", "MambaModel", "Model"])
      end

      if MlxLm::Models.const_defined?(:Mamba2, false)
        namespace = MlxLm::Models::Mamba2
        define_class(namespace, :Mamba2, ["MambaRMSNormGated", "Mamba2Block", "ResidualBlock", "Mamba2Model", "Model"])
      end

      if MlxLm::Models.const_defined?(:Minimax, false)
        namespace = MlxLm::Models::Minimax
        define_class(namespace, :MiniMaxAttention, ["Attention", "SparseMoeBlock", "DecoderLayer", "MiniMaxModel", "Model"])
        define_class(namespace, :MiniMaxDecoderLayer, ["DecoderLayer", "Attention", "SparseMoeBlock", "MiniMaxModel", "Model"])
        define_class(namespace, :MiniMaxSparseMoeBlock, ["SparseMoeBlock", "Attention", "DecoderLayer", "MiniMaxModel", "Model"])
        define_class(namespace, :ShardedRMSNorm, ["Attention", "SparseMoeBlock", "DecoderLayer", "MiniMaxModel", "Model"])
      end

      if MlxLm::Models.const_defined?(:Mixtral, false)
        namespace = MlxLm::Models::Mixtral
        define_class(namespace, :MixtralAttention, ["Attention", "SparseMoeBlock", "MixtralDecoderLayer", "MixtralModel", "Model"])
        define_class(namespace, :MixtralSparseMoeBlock, ["SparseMoeBlock", "Attention", "MixtralDecoderLayer", "MixtralModel", "Model"])
      end

      if MlxLm::Models.const_defined?(:NemotronH, false)
        namespace = MlxLm::Models::NemotronH
        define_class(namespace, :MambaRMSNormGated, ["Model"])
        define_class(namespace, :MoEGate, ["Model"])
        define_class(namespace, :NemotronHAttention, ["Model"])
        define_class(namespace, :NemotronHBlock, ["Model"])
        define_class(namespace, :NemotronHMLP, ["Model"])
        define_class(namespace, :NemotronHMamba2Mixer, ["Model"])
        define_class(namespace, :NemotronHMoE, ["Model"])
        define_class(namespace, :NemotronHModel, ["Model"])
      end

      if MlxLm::Models.const_defined?(:OLMo2, false)
        namespace = MlxLm::Models::OLMo2
        define_class(namespace, :LlamaModel, ["OLMo2Model", "Model", "Attention", "MLP", "TransformerBlock"])
      end

      if MlxLm::Models.const_defined?(:OLMoE, false)
        namespace = MlxLm::Models::OLMoE
        define_class(namespace, :Attention, ["Model"])
        define_class(namespace, :OlmoeModel, ["Model"])
        define_class(namespace, :OlmoeSparseMoeBlock, ["Model"])
        define_class(namespace, :TransformerBlock, ["Model"])
      end

      if MlxLm::Models.const_defined?(:PhiMoe, false)
        namespace = MlxLm::Models::PhiMoe
        define_class(namespace, :PhiMoEDecoderLayer, ["DecoderLayer", "Attention", "PhiMoESparseMoeBlock", "PhiMoEModel", "Model"])
      end

      if MlxLm::Models.const_defined?(:Plamo, false)
        namespace = MlxLm::Models::Plamo
        define_class(namespace, :PlamoDecoder, ["Attention", "MLP", "DecoderLayer", "PlamoModel", "Model"])
        define_class(namespace, :PlamoDecoderLayer, ["DecoderLayer", "Attention", "MLP", "PlamoModel", "Model"])
      end

      if MlxLm::Models.const_defined?(:Plamo2, false)
        namespace = MlxLm::Models::Plamo2
        define_class(namespace, :Attention, ["Model"])
        define_class(namespace, :MLP, ["Model"])
        define_class(namespace, :Mamba, ["Model"])
        define_class(namespace, :PlamoDecoder, ["Model"])
        define_class(namespace, :PlamoDecoderLayer, ["Model"])
        define_class(namespace, :PlamoModel, ["Model"])
        define_class(namespace, :RMSNorm, ["Model"])
      end

      if MlxLm::Models.const_defined?(:Qwen2Moe, false)
        namespace = MlxLm::Models::Qwen2Moe
        define_class(namespace, :Attention, ["SharedExpertMLP", "SparseMoeBlock", "DecoderLayer", "Qwen2MoeModel", "Model"])
        define_class(namespace, :MLP, ["SharedExpertMLP", "SparseMoeBlock", "DecoderLayer", "Qwen2MoeModel", "Model"])
        define_class(namespace, :Qwen2MoeDecoderLayer, ["DecoderLayer", "SharedExpertMLP", "SparseMoeBlock", "Qwen2MoeModel", "Model"])
        define_class(namespace, :Qwen2MoeSparseMoeBlock, ["SparseMoeBlock", "Qwen2MoeModel", "SharedExpertMLP", "DecoderLayer", "Model"])
      end

      if MlxLm::Models.const_defined?(:Qwen35, false)
        namespace = MlxLm::Models::Qwen35
        define_class(namespace, :DecoderLayer, ["Model"])
        define_class(namespace, :GatedDeltaNet, ["Model"])
        define_class(namespace, :Qwen3_5TextModel, ["Model"])
        define_class(namespace, :TextModel, ["Model"])
        define_class(namespace, :TextModelArgs, ["Model"])
      end

      if MlxLm::Models.const_defined?(:Qwen3Moe, false)
        namespace = MlxLm::Models::Qwen3Moe
        define_class(namespace, :Attention, ["SparseMoeBlock", "DecoderLayer", "Qwen3MoeModel", "Model"])
        define_class(namespace, :MLP, ["SparseMoeBlock", "DecoderLayer", "Qwen3MoeModel", "Model"])
        define_class(namespace, :Qwen3MoeDecoderLayer, ["DecoderLayer", "SparseMoeBlock", "Qwen3MoeModel", "Model"])
        define_class(namespace, :Qwen3MoeSparseMoeBlock, ["SparseMoeBlock", "Qwen3MoeModel", "DecoderLayer", "Model"])
      end

      if MlxLm::Models.const_defined?(:Qwen3Next, false)
        namespace = MlxLm::Models::Qwen3Next
        define_class(namespace, :Qwen3NextAttention, ["Model"])
        define_class(namespace, :Qwen3NextDecoderLayer, ["Model"])
        define_class(namespace, :Qwen3NextGatedDeltaNet, ["Model"])
        define_class(namespace, :Qwen3NextMLP, ["Model"])
        define_class(namespace, :Qwen3NextModel, ["Model"])
        define_class(namespace, :Qwen3NextRMSNormGated, ["Model"])
        define_class(namespace, :Qwen3NextSparseMoeBlock, ["Model"])
      end

      if MlxLm::Models.const_defined?(:RecurrentGemma, false)
        namespace = MlxLm::Models::RecurrentGemma
        define_class(namespace, :Conv1d, ["RMSNorm", "RGLRU", "RecurrentBlock", "LocalAttentionBlock", "MLPBlock", "ResidualBlock", "Griffin", "Model"])
      end

      if MlxLm::Models.const_defined?(:Rwkv7, false)
        namespace = MlxLm::Models::Rwkv7
        define_class(namespace, :LayerNormPerHead, ["Model"])
        define_class(namespace, :LoRA, ["Model"])
        define_class(namespace, :Rwkv7ChannelMixing, ["Model"])
        define_class(namespace, :Rwkv7Layer, ["Model"])
        define_class(namespace, :Rwkv7Model, ["Model"])
        define_class(namespace, :Rwkv7TimeMixing, ["Model"])
        define_class(namespace, :TokenShift, ["Model"])
      end

      if MlxLm::Models.const_defined?(:StableLM, false)
        namespace = MlxLm::Models::StableLM
        define_class(namespace, :LayerNormPerHead, ["Attention", "MLP", "DecoderLayer", "StableLMModel", "Model"])
        define_class(namespace, :StableLM, ["Attention", "MLP", "DecoderLayer", "StableLMModel", "Model"])
      end

      if MlxLm::Models.const_defined?(:Step3p5, false)
        namespace = MlxLm::Models::Step3p5
        define_class(namespace, :ClampedSwiGLU, ["ZeroCenteredRMSNorm", "Step3p5MLP", "Step3p5MoEGate", "Step3p5MoE", "Step3p5Attention", "Step3p5DecoderLayer", "Step3p5Model", "Model"])
      end

    end
  end
end

MlxLm::ParityAliases.apply!