#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cerrno>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__linux__)
#include <sched.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor_ptr.h>
#include <executorch/extension/threadpool/threadpool.h>
#include <executorch/runtime/platform/runtime.h>

namespace {

using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::aten::TensorShapeDynamism;
using executorch::extension::TensorPtr;
using executorch::extension::clone_tensor_ptr;
using executorch::extension::make_tensor_ptr;
using executorch::extension::module::Module;
using executorch::runtime::EValue;
using executorch::runtime::Error;

struct Options {
  std::filesystem::path artifacts_dir;
  std::filesystem::path output_file;
  std::string precision_label = "unspecified";
  int num_steps = 10;
  int cpu_threads = 5;
  int vision_cpu_threads = 0;
  int camera_count = 3;
  int image_size = 512;
  int language_length = 19;
  std::string cpu_affinity;
  std::string vision_cpu_affinity;
  int action_dim = 6;
  int warmup_runs = 0;
  int benchmark_runs = 1;

  bool uses_vision_worker() const {
    return vision_cpu_threads > 0 || !vision_cpu_affinity.empty();
  }
};

struct Inputs {
  TensorPtr images;
  TensorPtr image_masks;
  TensorPtr language_tokens;
  TensorPtr language_mask;
  TensorPtr state;
  TensorPtr actions;
};

struct Timings {
  double vision_ms;
  double vision_compute_ms;
  double prefix_ms;
  double denoise_ms;

  double total_ms() const {
    return vision_ms + prefix_ms + denoise_ms;
  }
};

struct InferenceResult {
  Timings timings;
  std::vector<float> actions;
};

struct Statistics {
  double mean;
  double standard_deviation;
  double median;
  double p95;
  double minimum;
  double maximum;
};

std::string option_value(const std::string& argument, const std::string& name) {
  const std::string prefix = name + "=";
  if (argument.rfind(prefix, 0) != 0) {
    return {};
  }
  return argument.substr(prefix.size());
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (const auto value = option_value(argument, "--artifacts_dir"); !value.empty()) {
      options.artifacts_dir = value;
    } else if (const auto value = option_value(argument, "--output_file"); !value.empty()) {
      options.output_file = value;
    } else if (const auto value = option_value(argument, "--precision_label"); !value.empty()) {
      options.precision_label = value;
    } else if (const auto value = option_value(argument, "--num_steps"); !value.empty()) {
      options.num_steps = std::stoi(value);
    } else if (const auto value = option_value(argument, "--cpu_threads"); !value.empty()) {
      options.cpu_threads = std::stoi(value);
    } else if (const auto value = option_value(argument, "--vision_cpu_threads"); !value.empty()) {
      options.vision_cpu_threads = std::stoi(value);
    } else if (const auto value = option_value(argument, "--camera_count"); !value.empty()) {
      options.camera_count = std::stoi(value);
    } else if (const auto value = option_value(argument, "--cpu_affinity"); !value.empty()) {
      options.cpu_affinity = value;
    } else if (const auto value = option_value(argument, "--vision_cpu_affinity"); !value.empty()) {
      options.vision_cpu_affinity = value;
    } else if (const auto value = option_value(argument, "--image_size"); !value.empty()) {
      options.image_size = std::stoi(value);
    } else if (const auto value = option_value(argument, "--language_length"); !value.empty()) {
      options.language_length = std::stoi(value);
    } else if (const auto value = option_value(argument, "--action_dim"); !value.empty()) {
      options.action_dim = std::stoi(value);
    } else if (const auto value = option_value(argument, "--warmup_runs"); !value.empty()) {
      options.warmup_runs = std::stoi(value);
    } else if (const auto value = option_value(argument, "--benchmark_runs"); !value.empty()) {
      options.benchmark_runs = std::stoi(value);
    } else {
      throw std::runtime_error("Unknown or malformed argument: " + argument);
    }
  }

  if (options.artifacts_dir.empty()) {
    throw std::runtime_error("--artifacts_dir is required");
  }
  if (options.output_file.empty()) {
    options.output_file = options.artifacts_dir / "native_orchestrator_output.bin";
  }
  if (options.num_steps < 1 || options.cpu_threads < 1 ||
      options.benchmark_runs < 1 || options.warmup_runs < 0) {
    throw std::runtime_error(
        "--num_steps, --cpu_threads, and --benchmark_runs must be at least 1; "
        "--warmup_runs must be non-negative");
  }
  if (options.camera_count < 1 || options.image_size < 1 ||
      options.language_length < 1) {
    throw std::runtime_error(
        "--camera_count, --image_size, and --language_length must be at least 1");
  }
  if (options.action_dim < 1 || options.action_dim > 32) {
    throw std::runtime_error("--action_dim must be between 1 and 32");
  }
  if (options.vision_cpu_threads < 0) {
    throw std::runtime_error("--vision_cpu_threads cannot be negative");
  }
  if (options.uses_vision_worker()) {
    if (options.vision_cpu_threads == 0) {
      options.vision_cpu_threads = options.cpu_threads;
    }
    if (options.vision_cpu_affinity.empty()) {
      options.vision_cpu_affinity = options.cpu_affinity;
    }
  }
  return options;
}

std::vector<int> parse_cpu_affinity(const std::string& specification) {
  std::vector<int> cpus;
  std::size_t offset = 0;
  while (offset < specification.size()) {
    const std::size_t comma = specification.find(',', offset);
    const std::size_t end =
        comma == std::string::npos ? specification.size() : comma;
    const std::string token = specification.substr(offset, end - offset);
    if (token.empty()) {
      throw std::runtime_error("Malformed --cpu_affinity: " + specification);
    }

    const std::size_t dash = token.find('-');
    const int first = std::stoi(token.substr(0, dash));
    const int last = dash == std::string::npos
        ? first
        : std::stoi(token.substr(dash + 1));
    if (first < 0 || last < first) {
      throw std::runtime_error("Malformed --cpu_affinity: " + specification);
    }
    for (int cpu = first; cpu <= last; ++cpu) {
#if defined(__linux__)
      if (cpu >= CPU_SETSIZE) {
        throw std::runtime_error(
            "CPU index exceeds CPU_SETSIZE in --cpu_affinity");
      }
#endif
      cpus.push_back(cpu);
    }

    if (comma == std::string::npos) {
      break;
    }
    offset = comma + 1;
  }
  if (cpus.empty()) {
    throw std::runtime_error("--cpu_affinity cannot be empty");
  }
  return cpus;
}

void apply_cpu_affinity(const std::string& specification) {
  if (specification.empty()) {
    return;
  }
#if defined(__linux__)
  cpu_set_t cpu_set;
  CPU_ZERO(&cpu_set);
  for (const int cpu : parse_cpu_affinity(specification)) {
    CPU_SET(cpu, &cpu_set);
  }
  if (::sched_setaffinity(0, sizeof(cpu_set), &cpu_set) != 0) {
    throw std::runtime_error(
        "sched_setaffinity failed for " + specification + ": " +
        std::strerror(errno));
  }
#else
  throw std::runtime_error("--cpu_affinity is supported only on Linux");
#endif
}

std::size_t numel(const std::vector<executorch::aten::SizesType>& sizes) {
  std::size_t count = 1;
  for (const auto size : sizes) {
    count *= static_cast<std::size_t>(size);
  }
  return count;
}

TensorPtr load_tensor(
    const std::filesystem::path& path,
    std::vector<executorch::aten::SizesType> sizes,
    ScalarType scalar_type,
    std::size_t element_size) {
  const std::size_t byte_count = numel(sizes) * element_size;
  const auto actual_size = std::filesystem::file_size(path);
  if (actual_size != byte_count) {
    throw std::runtime_error(
        path.string() + " has " + std::to_string(actual_size) +
        " bytes; expected " + std::to_string(byte_count));
  }

  void* data = ::operator new(byte_count);
  std::ifstream input(path, std::ios::binary);
  if (!input.read(static_cast<char*>(data), static_cast<std::streamsize>(byte_count))) {
    ::operator delete(data);
    throw std::runtime_error("Failed to read " + path.string());
  }

  return make_tensor_ptr(
      std::move(sizes),
      data,
      scalar_type,
      TensorShapeDynamism::STATIC,
      [](void* pointer) { ::operator delete(pointer); });
}

std::filesystem::path action_input_path(
    const std::filesystem::path& denoise_inputs) {
  const auto actions = denoise_inputs / "actions.bin";
  if (std::filesystem::exists(actions)) {
    return actions;
  }
  return denoise_inputs / "noise.bin";
}

Inputs load_inputs(const Options& options) {
  const auto native_inputs = options.artifacts_dir / "native_runner";
  return {
      load_tensor(
          native_inputs / "vision_encoder" / "images.bin",
          {1, options.camera_count, 3, options.image_size, options.image_size},
          ScalarType::Float,
          sizeof(float)),
      load_tensor(
          native_inputs / "prefix_forward" / "image_masks.bin",
          {1, options.camera_count},
          ScalarType::Bool,
          sizeof(std::uint8_t)),
      load_tensor(
          native_inputs / "prefix_forward" / "language_tokens.bin",
          {1, options.language_length},
          ScalarType::Long,
          sizeof(std::int64_t)),
      load_tensor(
          native_inputs / "prefix_forward" / "language_mask.bin",
          {1, options.language_length},
          ScalarType::Bool,
          sizeof(std::uint8_t)),
      load_tensor(
          native_inputs / "prefix_forward" / "state.bin",
          {1, 32},
          ScalarType::Float,
          sizeof(float)),
      load_tensor(
          action_input_path(native_inputs / "denoise_step"),
          {1, 50, 32},
          ScalarType::Float,
          sizeof(float)),
  };
}

std::vector<EValue> run_module(Module& module, std::vector<EValue> inputs) {
  auto result = module.forward(inputs);
  if (!result.ok()) {
    throw std::runtime_error(
        "ExecuTorch forward failed with status " +
        std::to_string(static_cast<std::uint32_t>(result.error())));
  }
  return std::move(result.get());
}

const Tensor& tensor_output(const std::vector<EValue>& outputs, std::size_t index) {
  if (index >= outputs.size() || !outputs[index].isTensor()) {
    throw std::runtime_error("Missing tensor output " + std::to_string(index));
  }
  return outputs[index].toTensor();
}

void load_forward(Module& module, const std::string& name) {
  const Error error = module.load_forward();
  if (error != Error::Ok) {
    throw std::runtime_error(
        "Failed to load " + name + " with status " +
        std::to_string(static_cast<std::uint32_t>(error)));
  }
}

double milliseconds(
    const std::chrono::steady_clock::time_point start,
    const std::chrono::steady_clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

struct VisionStageResult {
  TensorPtr image_embeddings;
  double wall_ms;
  double compute_ms;
};

#if defined(__linux__)

constexpr std::uint32_t kVisionPacketMagic = 0x534D564C;
constexpr std::size_t kMaxTensorRank = 8;

struct VisionPacketHeader {
  std::uint32_t magic;
  std::uint32_t rank;
  std::uint64_t numel;
  std::array<std::int64_t, kMaxTensorRank> sizes;
  double compute_ms;
};

void write_exact(const int descriptor, const void* data, std::size_t size) {
  const auto* cursor = static_cast<const std::uint8_t*>(data);
  while (size > 0) {
    const ssize_t written = ::write(descriptor, cursor, size);
    if (written < 0 && errno == EINTR) {
      continue;
    }
    if (written <= 0) {
      throw std::runtime_error(
          "Vision-worker pipe write failed: " +
          std::string(written < 0 ? std::strerror(errno) : "closed pipe"));
    }
    cursor += written;
    size -= static_cast<std::size_t>(written);
  }
}

void read_exact(const int descriptor, void* data, std::size_t size) {
  auto* cursor = static_cast<std::uint8_t*>(data);
  while (size > 0) {
    const ssize_t received = ::read(descriptor, cursor, size);
    if (received < 0 && errno == EINTR) {
      continue;
    }
    if (received <= 0) {
      throw std::runtime_error(
          "Vision-worker pipe read failed: " +
          std::string(received < 0 ? std::strerror(errno) : "worker exited"));
    }
    cursor += received;
    size -= static_cast<std::size_t>(received);
  }
}

class VisionWorker final {
 public:
  VisionWorker(const Options& options, const TensorPtr& images) {
    int commands[2];
    int results[2];
    if (::pipe(commands) != 0) {
      throw std::runtime_error(
          "Failed to create vision command pipe: " +
          std::string(std::strerror(errno)));
    }
    if (::pipe(results) != 0) {
      ::close(commands[0]);
      ::close(commands[1]);
      throw std::runtime_error(
          "Failed to create vision result pipe: " +
          std::string(std::strerror(errno)));
    }

    std::cout.flush();
    std::cerr.flush();
    process_id_ = ::fork();
    if (process_id_ < 0) {
      const std::string message = std::strerror(errno);
      ::close(commands[0]);
      ::close(commands[1]);
      ::close(results[0]);
      ::close(results[1]);
      throw std::runtime_error("Failed to fork vision worker: " + message);
    }
    if (process_id_ == 0) {
      ::close(commands[1]);
      ::close(results[0]);
      child_main(options, images, commands[0], results[1]);
    }

    command_descriptor_ = commands[1];
    result_descriptor_ = results[0];
    ::close(commands[0]);
    ::close(results[1]);
    std::signal(SIGPIPE, SIG_IGN);
    try {
      std::uint8_t ready = 0;
      read_exact(result_descriptor_, &ready, sizeof(ready));
      if (ready != 1) {
        throw std::runtime_error("Vision worker returned an invalid ready signal");
      }
    } catch (...) {
      shutdown();
      throw;
    }
  }

  VisionWorker(const VisionWorker&) = delete;
  VisionWorker& operator=(const VisionWorker&) = delete;

  ~VisionWorker() {
    shutdown();
  }

  VisionStageResult run() {
    const auto wall_start = std::chrono::steady_clock::now();
    const std::uint8_t command = 1;
    write_exact(command_descriptor_, &command, sizeof(command));

    VisionPacketHeader header{};
    read_exact(result_descriptor_, &header, sizeof(header));
    if (header.magic != kVisionPacketMagic || header.rank == 0 ||
        header.rank > kMaxTensorRank || header.numel == 0) {
      throw std::runtime_error("Vision worker returned invalid tensor metadata");
    }
    std::vector<executorch::aten::SizesType> sizes;
    sizes.reserve(header.rank);
    std::size_t expected_numel = 1;
    for (std::size_t index = 0; index < header.rank; ++index) {
      if (header.sizes[index] < 1) {
        throw std::runtime_error("Vision worker returned an invalid tensor shape");
      }
      sizes.push_back(
          static_cast<executorch::aten::SizesType>(header.sizes[index]));
      expected_numel *= static_cast<std::size_t>(header.sizes[index]);
    }
    if (expected_numel != header.numel) {
      throw std::runtime_error("Vision worker tensor size does not match its shape");
    }
    std::vector<float> data(header.numel);
    read_exact(
        result_descriptor_, data.data(), data.size() * sizeof(data.front()));
    const auto wall_end = std::chrono::steady_clock::now();
    return {
        make_tensor_ptr(
            std::move(sizes),
            std::move(data),
            {},
            {},
            ScalarType::Float,
            TensorShapeDynamism::STATIC),
        milliseconds(wall_start, wall_end),
        header.compute_ms,
    };
  }

 private:
  void shutdown() noexcept {
    if (command_descriptor_ >= 0) {
      const std::uint8_t quit = 0;
      while (::write(command_descriptor_, &quit, sizeof(quit)) < 0 &&
             errno == EINTR) {
      }
      ::close(command_descriptor_);
      command_descriptor_ = -1;
    }
    if (result_descriptor_ >= 0) {
      ::close(result_descriptor_);
      result_descriptor_ = -1;
    }
    if (process_id_ > 0) {
      int status = 0;
      while (::waitpid(process_id_, &status, 0) < 0 && errno == EINTR) {
      }
      process_id_ = -1;
    }
  }
  [[noreturn]] static void child_main(
      const Options& options,
      const TensorPtr& images,
      const int command_descriptor,
      const int result_descriptor) {
    try {
      apply_cpu_affinity(options.vision_cpu_affinity);
      executorch::runtime::runtime_init();
      executorch::extension::threadpool::get_threadpool()
          ->_unsafe_reset_threadpool(
              static_cast<std::uint32_t>(options.vision_cpu_threads));
      Module vision_module(
          (options.artifacts_dir / "vision_encoder_xnnpack.pte").string(),
          Module::LoadMode::Mmap);
      load_forward(vision_module, "vision_encoder");
      const std::uint8_t ready = 1;
      write_exact(result_descriptor, &ready, sizeof(ready));

      while (true) {
        std::uint8_t command = 0;
        read_exact(command_descriptor, &command, sizeof(command));
        if (command == 0) {
          break;
        }
        if (command != 1) {
          throw std::runtime_error("Vision worker received an invalid command");
        }
        const auto compute_start = std::chrono::steady_clock::now();
        const auto outputs = run_module(vision_module, {images});
        const Tensor& embeddings = tensor_output(outputs, 0);
        const auto compute_end = std::chrono::steady_clock::now();
        if (embeddings.scalar_type() != ScalarType::Float ||
            embeddings.dim() < 1 ||
            static_cast<std::size_t>(embeddings.dim()) > kMaxTensorRank) {
          throw std::runtime_error("Vision worker produced an unsupported tensor");
        }
        VisionPacketHeader header{};
        header.magic = kVisionPacketMagic;
        header.rank = static_cast<std::uint32_t>(embeddings.dim());
        header.numel = static_cast<std::uint64_t>(embeddings.numel());
        for (std::size_t index = 0; index < header.rank; ++index) {
          header.sizes[index] = embeddings.size(index);
        }
        header.compute_ms = milliseconds(compute_start, compute_end);
        write_exact(result_descriptor, &header, sizeof(header));
        write_exact(
            result_descriptor,
            embeddings.const_data_ptr<float>(),
            static_cast<std::size_t>(embeddings.nbytes()));
      }
      ::close(command_descriptor);
      ::close(result_descriptor);
      ::_exit(EXIT_SUCCESS);
    } catch (const std::exception& error) {
      std::cerr << "Vision worker error: " << error.what() << '\n';
      ::_exit(EXIT_FAILURE);
    }
  }

  pid_t process_id_ = -1;
  int command_descriptor_ = -1;
  int result_descriptor_ = -1;
};

#else

class VisionWorker final {
 public:
  VisionWorker(const Options&, const TensorPtr&) {
    throw std::runtime_error(
        "The separate vision runtime is supported only on Linux");
  }
  VisionStageResult run() {
    throw std::runtime_error(
        "The separate vision runtime is supported only on Linux");
  }
};

#endif

InferenceResult run_inference(
    Module* vision_module,
    VisionWorker* vision_worker,
    Module& prefix_module,
    Module& denoise_module,
    const Inputs& inputs,
    int num_steps,
    int action_dim) {
  std::vector<EValue> vision_outputs;
  TensorPtr worker_image_embeddings;
  const Tensor* image_embeddings = nullptr;
  double vision_ms = 0.0;
  double vision_compute_ms = 0.0;
  if (vision_worker != nullptr) {
    VisionStageResult result = vision_worker->run();
    worker_image_embeddings = std::move(result.image_embeddings);
    image_embeddings = worker_image_embeddings.get();
    vision_ms = result.wall_ms;
    vision_compute_ms = result.compute_ms;
  } else {
    if (vision_module == nullptr) {
      throw std::runtime_error("Vision module is unavailable");
    }
    const auto vision_start = std::chrono::steady_clock::now();
    vision_outputs = run_module(*vision_module, {inputs.images});
    image_embeddings = &tensor_output(vision_outputs, 0);
    const auto vision_end = std::chrono::steady_clock::now();
    vision_ms = milliseconds(vision_start, vision_end);
    vision_compute_ms = vision_ms;
  }

  const auto prefix_start = std::chrono::steady_clock::now();
  const auto prefix_outputs = run_module(
      prefix_module,
      {*image_embeddings,
       inputs.image_masks,
       inputs.language_tokens,
       inputs.language_mask,
       inputs.state});
  const Tensor& prefix_mask = tensor_output(prefix_outputs, 0);
  const Tensor& flat_cache = tensor_output(prefix_outputs, 1);
  const auto prefix_end = std::chrono::steady_clock::now();

  const auto denoise_start = std::chrono::steady_clock::now();
  TensorPtr current_actions = inputs.actions;
  std::vector<EValue> denoise_outputs;
  for (int step = 0; step < num_steps; ++step) {
    const float timestep_value =
        1.0f - static_cast<float>(step) / static_cast<float>(num_steps);
    auto timestep_tensor = make_tensor_ptr({1}, std::vector<float>{timestep_value});
    denoise_outputs = run_module(
        denoise_module,
        {prefix_mask,
         flat_cache,
         current_actions,
         timestep_tensor});
    const Tensor& updated_actions = tensor_output(denoise_outputs, 0);
    if (updated_actions.numel() != 50 * 32) {
      throw std::runtime_error("Unexpected denoise output size");
    }

    // A module's planned output buffer may be reused by its next invocation.
    // Copy only this small recurrent tensor; the much larger vision and KV-cache
    // outputs remain live in their EValue vectors without cloning.
    if (step + 1 < num_steps) {
      current_actions = clone_tensor_ptr(updated_actions);
    }
  }
  const auto denoise_end = std::chrono::steady_clock::now();

  const Tensor& actions = tensor_output(denoise_outputs, 0);
  const float* action_data = actions.const_data_ptr<float>();
  std::vector<float> cropped_actions;
  cropped_actions.reserve(50 * action_dim);
  for (int token = 0; token < 50; ++token) {
    const auto offset = static_cast<std::size_t>(token * 32);
    cropped_actions.insert(
        cropped_actions.end(),
        action_data + offset,
        action_data + offset + action_dim);
  }

  return {
      {vision_ms,
       vision_compute_ms,
       milliseconds(prefix_start, prefix_end),
       milliseconds(denoise_start, denoise_end)},
      std::move(cropped_actions),
  };
}

Statistics statistics(const std::vector<double>& values) {
  const double mean =
      std::accumulate(values.begin(), values.end(), 0.0) / values.size();
  double squared_difference_sum = 0.0;
  for (const double value : values) {
    const double difference = value - mean;
    squared_difference_sum += difference * difference;
  }
  const double standard_deviation = values.size() == 1
      ? 0.0
      : std::sqrt(squared_difference_sum / (values.size() - 1));

  std::vector<double> sorted = values;
  std::sort(sorted.begin(), sorted.end());
  const auto percentile = [&sorted](const double proportion) {
    const double position = proportion * static_cast<double>(sorted.size() - 1);
    const auto lower_index = static_cast<std::size_t>(std::floor(position));
    const auto upper_index = static_cast<std::size_t>(std::ceil(position));
    const double fraction = position - static_cast<double>(lower_index);
    return sorted[lower_index] * (1.0 - fraction) +
        sorted[upper_index] * fraction;
  };
  return {
      mean,
      standard_deviation,
      percentile(0.5),
      percentile(0.95),
      sorted.front(),
      sorted.back(),
  };
}

std::vector<double> timing_values(
    const std::vector<Timings>& timings,
    double Timings::*member) {
  std::vector<double> values;
  values.reserve(timings.size());
  for (const Timings& timing : timings) {
    values.push_back(timing.*member);
  }
  return values;
}

std::vector<double> total_values(const std::vector<Timings>& timings) {
  std::vector<double> values;
  values.reserve(timings.size());
  for (const Timings& timing : timings) {
    values.push_back(timing.total_ms());
  }
  return values;
}

void write_actions(
    const std::filesystem::path& output_path,
    const std::vector<float>& actions) {
  std::ofstream output(output_path, std::ios::binary);
  output.write(
      reinterpret_cast<const char*>(actions.data()),
      static_cast<std::streamsize>(actions.size() * sizeof(float)));
  if (!output) {
    throw std::runtime_error("Failed to write " + output_path.string());
  }
}

} // namespace

int main(int argc, char** argv) {
  try {
    // Progress is normally consumed through scripts/benchmark.py via a pipe.
    // Flush each insertion so warm-up and measured-run updates remain live.
    std::cout << std::unitbuf;
    const Options options = parse_options(argc, argv);
    const Inputs inputs = load_inputs(options);
    std::unique_ptr<VisionWorker> vision_worker;
    if (options.uses_vision_worker()) {
      // Fork before either process initializes ExecuTorch or creates worker
      // threads. Each XNNPACK runtime then keeps a valid, process-local pool.
      vision_worker = std::make_unique<VisionWorker>(options, inputs.images);
    }
    apply_cpu_affinity(options.cpu_affinity);
    executorch::runtime::runtime_init();
    executorch::extension::threadpool::get_threadpool()->_unsafe_reset_threadpool(
        static_cast<std::uint32_t>(options.cpu_threads));
    std::unique_ptr<Module> vision_module;
    if (!vision_worker) {
      vision_module = std::make_unique<Module>(
          (options.artifacts_dir / "vision_encoder_xnnpack.pte").string(),
          Module::LoadMode::Mmap);
    }
    Module prefix_module(
        (options.artifacts_dir / "prefix_forward_xnnpack.pte").string(),
        Module::LoadMode::Mmap);
    Module denoise_module(
        (options.artifacts_dir / "denoise_step_xnnpack.pte").string(),
        Module::LoadMode::Mmap);

    const auto load_start = std::chrono::steady_clock::now();
    if (vision_module) {
      load_forward(*vision_module, "vision_encoder");
    }
    load_forward(prefix_module, "prefix_forward");
    load_forward(denoise_module, "denoise_step");
    const auto load_end = std::chrono::steady_clock::now();
    std::cout << "Loaded " << (vision_worker ? "parent" : "all")
              << " methods in " << milliseconds(load_start, load_end)
              << " ms\n";
    std::cout << "Benchmark configuration: precision="
              << options.precision_label << ", cameras="
              << options.camera_count << ", threads="
              << options.cpu_threads << ", affinity="
              << (options.cpu_affinity.empty() ? "unrestricted" : options.cpu_affinity)
              << ", vision_runtime="
              << (vision_worker ? "separate" : "shared")
              << ", vision_threads="
              << (vision_worker ? options.vision_cpu_threads : options.cpu_threads)
              << ", vision_affinity="
              << (vision_worker
                      ? (options.vision_cpu_affinity.empty()
                             ? "unrestricted"
                             : options.vision_cpu_affinity)
                      : (options.cpu_affinity.empty()
                             ? "unrestricted"
                             : options.cpu_affinity))
              << ", image_size=" << options.image_size
              << ", language_length=" << options.language_length
              << ", denoising_steps=" << options.num_steps
              << ", warmup_runs=" << options.warmup_runs
              << ", benchmark_runs=" << options.benchmark_runs << "\n";

    for (int run_index = 0; run_index < options.warmup_runs; ++run_index) {
      std::cout << "Warm-up " << run_index + 1 << '/' << options.warmup_runs
                << "..." << std::flush;
      const auto result = run_inference(
          vision_module.get(),
          vision_worker.get(),
          prefix_module,
          denoise_module,
          inputs,
          options.num_steps,
          options.action_dim);
      std::cout << " " << result.timings.total_ms() << " ms" << std::endl;
    }

    std::vector<Timings> measured_timings;
    measured_timings.reserve(options.benchmark_runs);
    std::vector<float> final_actions;
    for (int run_index = 0; run_index < options.benchmark_runs; ++run_index) {
      const auto result = run_inference(
          vision_module.get(),
          vision_worker.get(),
          prefix_module,
          denoise_module,
          inputs,
          options.num_steps,
          options.action_dim);
      measured_timings.push_back(result.timings);
      final_actions = result.actions;
      std::cout << "Measured " << run_index + 1 << '/'
                << options.benchmark_runs << ": "
                << result.timings.total_ms() << " ms" << std::endl;
    }

    const Statistics vision = statistics(
        timing_values(measured_timings, &Timings::vision_ms));
    const Statistics vision_compute = statistics(
        timing_values(measured_timings, &Timings::vision_compute_ms));
    const Statistics prefix = statistics(
        timing_values(measured_timings, &Timings::prefix_ms));
    const Statistics denoise = statistics(
        timing_values(measured_timings, &Timings::denoise_ms));
    const Statistics total = statistics(total_values(measured_timings));

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Benchmark results (milliseconds):\n";
    const auto print_statistics = [](const char* label,
                                     const Statistics& result,
                                     const double divisor) {
      std::cout << label << ": mean " << result.mean / divisor
                << " +/- " << result.standard_deviation / divisor
                << ", median " << result.median / divisor
                << ", p95 " << result.p95 / divisor
                << ", min " << result.minimum / divisor
                << ", max " << result.maximum / divisor << "\n";
    };
    const std::string vision_label =
        "Vision, " + std::to_string(options.camera_count) + " cameras";
    print_statistics(vision_label.c_str(), vision, 1.0);
    if (vision_worker) {
      print_statistics("Vision worker compute", vision_compute, 1.0);
      std::vector<double> vision_ipc;
      vision_ipc.reserve(measured_timings.size());
      for (const Timings& timing : measured_timings) {
        vision_ipc.push_back(timing.vision_ms - timing.vision_compute_ms);
      }
      print_statistics("Vision IPC overhead", statistics(vision_ipc), 1.0);
    }
    print_statistics("Prefix", prefix, 1.0);
    print_statistics("Denoise, all steps", denoise, 1.0);
    print_statistics(
        "Denoise, per-step average",
        denoise,
        static_cast<double>(options.num_steps));
    print_statistics("Total inference", total, 1.0);

    write_actions(options.output_file, final_actions);
    std::cout << "Saved output to " << options.output_file << "\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << "\n";
    return EXIT_FAILURE;
  }
}
