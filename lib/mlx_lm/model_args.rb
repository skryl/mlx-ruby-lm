module MlxLm
  # Base class for model configuration arguments.
  # Mirrors Python's BaseModelArgs dataclass with from_dict filtering.
  class BaseModelArgs
    def self.fields
      @fields ||= {}
    end

    def self.field(name, default: :__required__)
      fields[name] = default

      attr_accessor name

      if default == :__required__
        # no default
      else
        define_method(:"default_#{name}") { default }
      end
    end

    def self.inherited(subclass)
      super
      # Copy parent fields into subclass
      subclass.instance_variable_set(:@fields, fields.dup)
    end

    def self.from_dict(params)
      known = {}
      fields.each do |name, default|
        str_name = name.to_s
        if params.key?(str_name)
          known[name] = params[str_name]
        elsif default != :__required__
          known[name] = default
        end
      end
      new(**known)
    end

    def initialize(**kwargs)
      kwargs.each do |k, v|
        if self.class.fields.key?(k)
          instance_variable_set(:"@#{k}", v)
        end
      end
      # Set defaults for any fields not provided
      self.class.fields.each do |name, default|
        next if kwargs.key?(name)
        next if default == :__required__
        instance_variable_set(:"@#{name}", default)
      end
    end
  end
end
