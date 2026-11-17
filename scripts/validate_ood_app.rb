#!/usr/bin/env ruby
# frozen_string_literal: true

require "digest"
require "erb"
require "open3"
require "ostruct"
require "pathname"
require "yaml"

ROOT = Pathname(__dir__).join("..").expand_path
REQUIRED = %w[
  manifest.yml form.yml.erb submit.yml.erb info.html.erb view.html.erb completed.html.erb
  template/script.sh.erb template/after.sh.erb template/connection.yml.erb template/.env
  template/bioemu2_colabfold.py template/build_report.py template/run_alphafold.py
  template/bioemu2_weights_lib.sh
].freeze
RUNNER_SHA256 = "70a1102908353d75bccfccfa85738f72d053a8fa819554c5e02e6230cc9ba86b"

class TemplateContext
  attr_reader :context

  def initialize
    @context = OpenStruct.new(
      run_name: "ci_run",
      protein_input: "ACDEFGHIKLMNPQRSTVWY",
      dna_input: "",
      rna_input: "",
      ligand_ccd: "ATP",
      ligand_smiles: "",
      bioemu2_mode: "structure",
      msa_mode: "single_sequence",
      seeds: "1,2",
      num_recycles: 10,
      num_diffusion_samples: 5,
      sampler: "edm",
      physicality: "high_quality",
      propagate_from: "",
      propagation_steps: 1,
      lag_ns: "1",
      weights_precision: "int8"
    )
  end

  def get_binding
    run_name = "ci_run"
    email_on_terminated = false
    user = "ci"
    context = @context
    binding
  end
end

missing = REQUIRED.reject { |path| ROOT.join(path).exist? }
abort("Missing required files:\n- #{missing.join("\n- ")}") unless missing.empty?

begin
  YAML.safe_load_file(ROOT.join("manifest.yml"), aliases: true)
rescue StandardError => e
  abort("manifest.yml is invalid: #{e.message}")
end

ctx = TemplateContext.new
Dir.glob(ROOT.join("**/*.erb")).sort.each do |path|
  begin
    RubyVM::InstructionSequence.compile(ERB.new(File.read(path)).src, path)
  rescue StandardError => e
    abort("ERB compilation failed for #{path}: #{e.message}")
  end
end

%w[form.yml.erb submit.yml.erb].each do |relative|
  begin
    rendered = ERB.new(File.read(ROOT.join(relative)), trim_mode: "-").result(ctx.get_binding)
    YAML.safe_load(rendered, aliases: true)
  rescue StandardError => e
    abort("Rendered YAML failed for #{relative}: #{e.message}")
  end
end

Dir.glob(ROOT.join("template/*.sh.erb")).sort.each do |path|
  rendered = ERB.new(File.read(path), trim_mode: "-").result(ctx.get_binding)
  _stdout, stderr, status = Open3.capture3("bash", "-n", stdin_data: rendered)
  abort("Shell validation failed for #{path}: #{stderr}") unless status.success?
end
Dir.glob(ROOT.join("scripts/*.sh")).sort.each do |path|
  _stdout, stderr, status = Open3.capture3("bash", "-n", path)
  abort("Shell validation failed for #{path}: #{stderr}") unless status.success?
end
# Plain shell helpers that live in template/ are staged into every OOD session
# and must be syntax-checked too (the *.sh.erb glob above does not catch them).
Dir.glob(ROOT.join("template/*.sh")).sort.each do |path|
  _stdout, stderr, status = Open3.capture3("bash", "-n", path)
  abort("Shell validation failed for #{path}: #{stderr}") unless status.success?
end
Dir.glob(ROOT.join("template/*.py")).sort.each do |path|
  source = File.read(path)
  _stdout, stderr, status = Open3.capture3(
    "python3", "-c", "import sys; compile(sys.stdin.read(), sys.argv[1], 'exec')", path,
    stdin_data: source
  )
  abort("Python validation failed for #{path}: #{stderr}") unless status.success?
end

actual = Digest::SHA256.file(ROOT.join("template/run_alphafold.py")).hexdigest
abort("Pinned run_alphafold.py checksum changed: #{actual}") unless actual == RUNNER_SHA256
puts "Open OnDemand BioEmu2 app validation passed"
