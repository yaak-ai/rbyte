{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
  };

  outputs =
    inputs@{ flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [
        "x86_64-linux"
        "aarch64-darwin"
      ];

      perSystem =
        { pkgs, ... }:
        let
          uvLibraryPath = pkgs.lib.makeLibraryPath [
            pkgs.stdenv.cc.cc.lib
            pkgs.zlib
            pkgs.openssl
            pkgs.libffi
            pkgs.ffmpeg-headless
          ];

          uvLibraryPathExport =
            if pkgs.stdenv.isDarwin then
              ''export DYLD_FALLBACK_LIBRARY_PATH="${uvLibraryPath}''${DYLD_FALLBACK_LIBRARY_PATH:+:''${DYLD_FALLBACK_LIBRARY_PATH}}"''
            else
              ''export LD_LIBRARY_PATH="${uvLibraryPath}''${NIX_LD_LIBRARY_PATH:+:''${NIX_LD_LIBRARY_PATH}}"'';

          uvWrapped = pkgs.writeShellScriptBin "uv" ''
            ${uvLibraryPathExport}
            exec ${pkgs.lib.getExe pkgs.uv} "$@"
          '';
        in
        {
          devShells.default = pkgs.mkShell {
            packages = with pkgs; [
              ffmpeg-headless
              gh
              git
              git-lfs
              just
              nushell
              openssh
              prek
              python312
              uvWrapped
              ytt

              # for `ymmv` and `m2df` via maturin
              cargo
              rustc
            ];
            env = {
              MATURIN_NO_INSTALL_RUST = "1";
              PYO3_PYTHON = pkgs.lib.getExe pkgs.python312;
              UV_NO_MANAGED_PYTHON = "1";
            };
          };
        };
    };
}
