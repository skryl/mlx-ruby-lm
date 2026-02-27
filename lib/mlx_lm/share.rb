require "pathname"

module MlxLm
  class DirectoryEntry
    include Comparable

    ENTRY_TYPE_ORDER = {
      "directory" => 0,
      "symlink" => 1,
      "file" => 2,
    }.freeze

    attr_reader :entry_type, :path, :dst

    def initialize(entry_type, path, dst = nil)
      unless ENTRY_TYPE_ORDER.key?(entry_type)
        raise ArgumentError, "unsupported entry_type: #{entry_type.inspect}"
      end

      @entry_type = entry_type
      @path = path.to_s
      @dst = dst&.to_s
    end

    def <=>(other)
      left = ENTRY_TYPE_ORDER.fetch(entry_type)
      right = ENTRY_TYPE_ORDER.fetch(other.entry_type)
      return left <=> right if left != right

      path <=> other.path
    end

    def ==(other)
      other.is_a?(DirectoryEntry) &&
        entry_type == other.entry_type &&
        path == other.path &&
        dst == other.dst
    end

    def self.from_path(root, path)
      root = Pathname.new(root)
      path = Pathname.new(path)

      entry_type = if path.symlink?
        "symlink"
      elsif path.directory?
        "directory"
      else
        "file"
      end
      dst = path.symlink? ? path.readlink.to_s : nil

      new(entry_type, path.relative_path_from(root).to_s, dst)
    end
  end
end
