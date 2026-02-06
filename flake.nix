{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable?shallow=1";
    flake-utils.url = "github:numtide/flake-utils?shallow=1";
  };

  outputs =
    { nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        devShells.default =
          with pkgs;
          mkShell {
            packages = [
              nushell
              git
              git-lfs
              uv
              ytt
              just
              skim
              ffmpeg-headless
            ];

            shellHook = lib.strings.concatLines [
              (lib.optionalString stdenv.isDarwin "export DYLD_FALLBACK_LIBRARY_PATH=${
                lib.makeLibraryPath [ ffmpeg-headless ]
              }")
            ];
          };
      }
    );
}
