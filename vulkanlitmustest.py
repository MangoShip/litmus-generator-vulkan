import litmustest

class VulkanLitmusTest(litmustest.LitmusTest):

    opencl_stress_mem_location = "scratchpad[scratch_locations[get_group_id(0)]]"
    # returns the first access in the stress pattern
    openCL_stress_first_access = {
        "store": ["{} = i;".format(opencl_stress_mem_location)],
        "load": ["uint tmp1 = {};".format(opencl_stress_mem_location),
            "if (tmp1 > 100) {", "  break;",
            "}"]
    }
    # given a first access, returns the second access in the stress pattern
    openCL_stress_second_access = {
        "store": {
            "store": ["{} = i + 1;".format(opencl_stress_mem_location)],
            "load": ["uint tmp1 = {};".format(opencl_stress_mem_location),
                "if (tmp1 > 100) {", "  break;",
                "}"]
        },
        "load": {
            "store": ["{} = i;".format(opencl_stress_mem_location)],
            "load": ["uint tmp2 = {};".format(opencl_stress_mem_location),
                "if (tmp2 > 100) {", "  break;",
                "}"]
        }
    }

    openCL_mem_order = {
            "relaxed": "memory_order_relaxed",
            "sc": "memory_order_seq_cst",
            "acquire": "memory_order_acquire",
            "release": "memory_order_release",
            "acq_rel": "memory_order_acq_rel"
        }

    # Code below this line generates the actual opencl kernel

    def file_ext(self):
        return ".cl"

    def generate_mem_loc(self, mem_loc, i, offset, should_shift, workgroup_id="shuffled_workgroup", use_local_id=False):
        shift_mem_loc = ""
        if should_shift:
            shift_mem_loc = "{} * get_local_size(0) + ".format(workgroup_id)
        if offset == 0:
            base = "{}id_{}".format(shift_mem_loc, i)
            offset_template = ""
        else:
            if use_local_id:
                to_permute = "get_local_id(0)"
            else:
                to_permute = "id_{}".format(i)
            base = "{}permute_id({}, stress_params[8], total_ids)".format(shift_mem_loc, to_permute)
            if offset == 1:
                offset_template = " + stress_params[11]"
            else:
                offset_template = " + {} * stress_params[11]".format(offset)
        return "uint {}_{} = ({}) * stress_params[10] * 2{};".format(mem_loc, i, base, offset_template)

    def generate_threads_header(self, test_mem_locs):
        new_local_id = "permute_id(get_local_id(0), stress_params[7], get_local_size(0))"
        suffix = []
        if len(self.threads) > 1:
            if self.same_workgroup:
                suffix = ["uint id_1 = {}".format(new_local_id)]
            else:
                suffix = [
                    "uint new_workgroup = stripe_workgroup(shuffled_workgroup, get_local_id(0), stress_params[9]);",
                    "uint id_1 = new_workgroup * get_local_size(0) + {};".format(new_local_id)
                ]
        if self.same_workgroup:
            prefix = [
                "uint total_ids = get_local_size(0);",
                "uint id_0 = get_local_id(0);"
            ]
            spin = "  spin(barrier, get_local_size(0));"
        else:
            prefix = [
                "uint total_ids = get_local_size(0) * stress_params[9];",
                "uint id_0 = shuffled_workgroup * get_local_size(0) + get_local_id(0);"
            ]
            spin = "  spin(barrier, get_local_size(0) * stress_params[9]);"
        statements = [
            "if (stress_params[4]) {",
            "  do_stress(scratchpad, scratch_locations, stress_params[5], stress_params[6]);",
            "}",
            "if (stress_params[0]) {",
            spin,
            "}"
        ]
        return prefix + suffix + test_mem_locs + statements

    def generate_helper_fns(self):
        permute_fn = [
            "static uint permute_id(uint id, uint factor, uint mask) {",
            "  return (id * factor) % mask;",
            "}",
            ""
        ]
        stripe_fn = [
            "static uint stripe_workgroup(uint workgroup_id, uint local_id, uint testing_workgroups) {",
            "  return (workgroup_id + 1 + local_id % (testing_workgroups - 1)) % testing_workgroups;",
            "}"
        ]
        return "\n".join(permute_fn + stripe_fn)

    def generate_meta(self):
        return "\n\n".join([self.generate_helper_fns()])

    def generate_stress(self):
        body = ["static void do_stress(__global uint* scratchpad, __global uint* scratch_locations, uint iterations, uint pattern) {",
                "for (uint i = 0; i < iterations; i++) {"]
        i = 0
        for first in self.openCL_stress_first_access:
            for second in self.openCL_stress_second_access[first]:
                if i == 0:
                    body += ["  if (pattern == 0) {"]
                else:
                    body += ["  }} else if (pattern == {}) {{".format(i)]
                body += ["    {}".format(statement) for statement in self.openCL_stress_first_access[first]]
                body += ["    {}".format(statement) for statement in self.openCL_stress_second_access[first][second]]
                i += 1
        body += ["  }", "}"]
        return "\n".join(["\n  ".join(body), "}"])

    def generate_spin(self):
        header = "static void spin(__global atomic_uint* barrier, uint limit) {"
        body = "\n  ".join([
            header,
            "int i = 0;",
            "uint val = atomic_fetch_add_explicit(barrier, 1, memory_order_relaxed);",
            "while (i < 1024 && val < limit) {",
            "  val = atomic_load_explicit(barrier, memory_order_relaxed);",
            "  i++;",
            "}"
        ])
        return "\n".join([body, "}"])

    def read_repr(self, instr, i):
        # set up rmw
        template = "uint {} = atomic_load_explicit(&test_locations[{}_{}], {});"
        return template.format(instr.variable, instr.mem_loc, i, self.openCL_mem_order[instr.mem_order])

    def write_repr(self, instr, i):
        # set up rmw
        template = "atomic_store_explicit(&test_locations[{}_{}], {}, {});"
        return template.format(instr.mem_loc, i, instr.value, self.openCL_mem_order[instr.mem_order])

    def fence_repr(self, instr):
        # Should fence have memory_order_seq_cst?
        return "atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_seq_cst, memory_scope_device);"

    def results_repr(self, variable, i):
        if variable == "r0":
            result_template = ""
        else:
            result_template = " + 1"
        return "atomic_store(&read_results[id_{}*2{}], {});".format(i, result_template, variable)

    # Find purpose of this 
    def thread_filter(self, first_thread, workgroup, thread):
        if first_thread:
            start = "if"
        else:
            start = "} else if"
        return start + " (shuffled_ids[get_global_id(0)] == get_local_size(0) * {} + {}) {{".format(workgroup, thread)

    def generate_stress_call(self):
        return [
            "  } else if (stress_params[1]) {",
            "    do_stress(scratchpad, scratch_locations, stress_params[2], stress_params[3]);",
            "  }"
        ]

    def generate_shader_def(self):
        return "\n".join([
            "__kernel void litmus_test (",
            "  __global atomic_uint* test_locations,",
            "  __global atomic_uint* read_results,",
            "  __global uint* shuffled_workgroups,",
            "  __global atomic_uint* barrier,",
            "  __global uint* scratchpad,",
            "  __global uint* scratch_locations,",
            "  __global uint* stress_params) {",
            "  uint shuffled_workgroup = shuffled_workgroups[get_group_id(0)];",
            "  if(shuffled_workgroup < stress_params[9]) {"
        ])

    def generate_result_storage(self):
        statements = ["atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_seq_cst, memory_scope_device);"]
        seen_ids = set()
        for behavior in self.behaviors:
            statements += self.generate_post_condition_stores(behavior.post_condition, seen_ids)
        return statements

    def generate_post_condition_stores(self, condition, seen_ids):
        result = []
        shift_mem_loc = "shuffled_workgroup * get_local_size(0)"
        if isinstance(condition, self.PostConditionLeaf):
            if condition.identifier not in seen_ids:
                seen_ids.add(condition.identifier)
                if condition.output_type == "variable":
                    variable = condition.identifier
                    if self.same_workgroup:
                        shift = "{} + ".format(shift_mem_loc)
                    else:
                        shift = ""
                    if variable == "r0":
                        result_template = ""
                    else:
                        result_template = " + 1"
                    result.append("atomic_store(&read_results[{}id_{}*2{}], {});".format(shift, self.read_threads[variable], result_template, variable))
                elif condition.output_type == "memory" and self.workgroup_memory:
                    mem_loc = "{}_{}".format(condition.identifier, len(self.threads) - 1)
                    # wg_test_locations?
                    result.append("atomic_store(&test_locations[{} * stress_params[10] * 2 + {}], atomic_load(&wg_test_locations[{}]));".format(shift_mem_loc, mem_loc, mem_loc))
        elif isinstance(condition, self.PostConditionNode):
            for cond in condition.conditions:
                result += self.generate_post_condition_stores(cond, seen_ids)
        return result
