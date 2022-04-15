# GPU Litmus Test Generator

Litmus tests are short concurrent programs that reveal relaxed behaviors in weak memory models and can be used to empirically test those models. Modern GPUs and their programming frameworks expose a weak memory model, and this tool provides a way to explore and understand the real world behaviors of these models. Currently, backends for [Vulkan](https://www.vulkan.org/) and [WebGPU](https://web.dev/gpu/) are supported.

## Test Generation
A litmus test starts out as a json configuration file which defines the set of actions each thread takes and outcomes that correspond to sequential, interleaved, or weak behaviors. The library of litmus tests is stored in the `litmus-config` directory. The output of the test generation process is a shader in the target backend's language, and is written into the `target` directory. Generallly, a shader consists of a set of testing threads which all run the litmus test in parallel and a set of stressing threads that are used to add stress to the memory system. Several other parameters are also included in the shader, so care must be taken to set up the shader inputs correctly. For more information on running the generated shaders, see the "Running a Test" section.

Along with the actual test shader, the test generator can optionally generate a results aggregation shader, which is used to aggregate all the behaviors seen by the parallel testing threads. While behaviors could be aggregated on the CPU side, adding this shader to the process makes testing much faster, and frameworks that run tests generated by this tool are encouraged to use it. To include the result aggregation shader in any test generation, add the flag `--gen_result_shader` to the test generation command.

### WebGPU
WebGPU's shading language is called WGSL. This tool supports WGSL shader generation out of the box, with no additional dependencies. To generate a WGSL shader, run `python3 litmusgenerator.py --backend wgsl <path-to-config-file>`.

### OpenCL (WIP)
OpenCL kernels are supported with no additional dependencies. To generate an OpenCL shader, run `python3 litmusgenerator.py --backend opencl <path-to-config-file>`.

### Vulkan (WIP)
Vulkan's shading language is called SPIR-V. Since SPIR-V is an intermediate representation, this tool does not generate SPIR-V directly. Instead, the tool first generates an OpenCL kernel and uses [clspv](https://github.com/google/clspv) to compile to SPIR-V. Therefore, for Vulkan test generation to work correctly clspv must be available as an executable on your system. To generate a SPIR-V shader, run `python3 litmusgenerator.py --backend vulkan <path-to-config-file>`.

## Running a Test
After generating your shaders, you probably want to run them. Depending on the target backend, the process for doing so will differ. To understand more about how the shaders actually work and their motivation, see [this](https://docs.google.com/presentation/d/1Gr8zbiE8yfBaijAqniv_rJbvs_mJzKapvfBoHkTZNjQ/edit) presentation.

### WebGPU
Generated WGSL shaders are currently run using https://gpuharbor.ucsc.edu/webgpu-mem-testing/. However, you are welcome to use them in your own custom backend for WebGPU.

### OpenCL
We do not currently have a way to directly run OpenCL kernels.

### Vulkan
The `test-runner` directory includes a C++ setup for running generated SPIR-V shaders. In order to use this, ensure you have cmake and Vulkan installed on your system, then follow the usual process to build an executable. The executable takes as input the test shader, the result shader, and a parameter file. Note that the test runner has a dependency on https://github.com/reeselevine/easyvk, which should be instantiated as a git submodule.

This process for running SPIR-V shaders on Vulkan is still a work in progress, so these instructions should get updated and include more detail as work continues.