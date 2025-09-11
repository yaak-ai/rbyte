{
  inputs.nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  inputs.flake-utils.url = "github:numtide/flake-utils";

  outputs = {
    nixpkgs,
    flake-utils,
    ...
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = nixpkgs.legacyPackages.${system};
      in {
        devShells.default = with pkgs;
          mkShell {
            packages =
              [
                git
                git-lfs
                uv
                ytt
                just
              ]
              ++ lib.optional stdenv.isDarwin [ffmpeg];

            shellHook = lib.strings.concatLines [
              "export PYTHONBREAKPOINT='pudb.set_trace'"
              (lib.optionalString
                stdenv.isDarwin
                "export DYLD_FALLBACK_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [pkgs.ffmpeg]}")
            ];
          };
      }
    );
}
