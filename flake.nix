{
  description = "ScopePack - Token-efficient context management for Claude Code";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        pythonPackages = pkgs.python311Packages;

        scopepack = pythonPackages.buildPythonPackage {
          pname = "scopepack";
          version = "0.1.0";
          format = "pyproject";

          src = ./.;

          nativeBuildInputs = with pythonPackages; [
            hatchling
          ];

          propagatedBuildInputs = with pythonPackages; [
            sentence-transformers
            transformers
            torch
            fastapi
            uvicorn
            httpx
            aiosqlite
            pydantic
            numpy
            # AST-based chunking (tree-sitter)
            tree-sitter-language-pack
          ];

          # Skip tests during build (they require models)
          doCheck = false;

          meta = with pkgs.lib; {
            description = "Token-efficient context management for Claude Code";
            homepage = "https://github.com/zach-source/context-scope-pack";
            license = licenses.mit;
          };
        };
      in
      {
        packages = {
          default = scopepack;
          scopepack = scopepack;
        };

        # Development shell (use devenv for full development)
        devShells.default = pkgs.mkShell {
          packages = [
            (pkgs.python311.withPackages (ps: [
              scopepack
              ps.pytest
              ps.pytest-asyncio
              ps.ruff
              ps.mypy
            ]))
          ];
        };
      }
    )
    // {
      # Home Manager module (system-independent)
      homeManagerModules = {
        default = self.homeManagerModules.scopepack;
        scopepack = import ./nix/home-module.nix self;
      };

      # Overlay for use in other flakes
      overlays.default = final: prev: {
        scopepack = self.packages.${prev.system}.scopepack;
      };
    };
}
