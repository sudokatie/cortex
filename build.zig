const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Create the main module
    const main_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Create cortex library module
    const cortex_mod = b.createModule(.{
        .root_source_file = b.path("src/cortex.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Add cortex as import to main
    main_mod.addImport("cortex", cortex_mod);

    // Executable
    const exe = b.addExecutable(.{
        .name = "cortex",
        .root_module = main_mod,
    });
    b.installArtifact(exe);

    // Run command
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }
    const run_step = b.step("run", "Run the cortex CLI");
    run_step.dependOn(&run_cmd.step);

    // Unit tests for library
    const lib_test_mod = b.createModule(.{
        .root_source_file = b.path("src/cortex.zig"),
        .target = target,
        .optimize = optimize,
    });
    const lib_unit_tests = b.addTest(.{
        .root_module = lib_test_mod,
    });
    const run_lib_unit_tests = b.addRunArtifact(lib_unit_tests);

    // Unit tests for main
    const main_test_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    main_test_mod.addImport("cortex", cortex_mod);
    const exe_unit_tests = b.addTest(.{
        .root_module = main_test_mod,
    });
    const run_exe_unit_tests = b.addRunArtifact(exe_unit_tests);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_lib_unit_tests.step);
    test_step.dependOn(&run_exe_unit_tests.step);
}
