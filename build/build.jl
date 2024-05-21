using Pkg
Pkg.add("PackageCompiler")

using PackageCompiler

Pkg.activate(".")
create_app(".", "artifact")
