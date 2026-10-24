// 编译 C 垫片，链接仓库里已经编好的 llama.cpp 动态库，并写入 rpath。
use std::path::PathBuf;

fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../third_party/llama.cpp");
    let root = root.canonicalize().expect("third_party/llama.cpp 不存在：先在 kdn-demo 下 make");
    let lib = root.join("build/bin");
    cc::Build::new()
        .file("csrc/shim.c")
        .include(root.join("include"))
        .include(root.join("ggml/include"))
        .opt_level(2)
        .compile("kdnshim");
    println!("cargo:rerun-if-changed=csrc/shim.c");
    println!("cargo:rustc-link-search=native={}", lib.display());
    for l in ["llama", "ggml", "ggml-base"] {
        println!("cargo:rustc-link-lib=dylib={l}");
    }
    // 绝对路径 rpath（本机原型），外加相对 target/release 的 rpath
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib.display());
    println!("cargo:rustc-link-arg=-Wl,-rpath,@executable_path/../../../third_party/llama.cpp/build/bin");
}
