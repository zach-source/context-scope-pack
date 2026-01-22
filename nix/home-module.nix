flake:
{
  config,
  lib,
  pkgs,
  ...
}:
let
  cfg = config.programs.scopepack;
  inherit (lib)
    mkEnableOption
    mkOption
    mkIf
    types
    ;

  # Get the scopepack package for this system
  scopepackPkg = flake.packages.${pkgs.system}.scopepack;

  # Hook files from the flake source
  hooksDir = "${flake}/.claude/hooks";
in
{
  options.programs.scopepack = {
    enable = mkEnableOption "ScopePack context compression for Claude Code";

    package = mkOption {
      type = types.package;
      default = scopepackPkg;
      description = "The scopepack package to use";
    };

    daemon = {
      enable = mkOption {
        type = types.bool;
        default = true;
        description = "Whether to run the scopepack daemon as a service";
      };

      port = mkOption {
        type = types.port;
        default = 18765;
        description = "Port for the daemon to listen on";
      };

      host = mkOption {
        type = types.str;
        default = "127.0.0.1";
        description = "Host for the daemon to bind to";
      };
    };

    hooks = {
      enable = mkOption {
        type = types.bool;
        default = true;
        description = "Whether to install Claude Code hooks";
      };

      preToolRead = mkOption {
        type = types.bool;
        default = true;
        description = "Enable PreToolUse(Read) hook for compressing large files";
      };

      postToolWriteEdit = mkOption {
        type = types.bool;
        default = true;
        description = "Enable PostToolUse(Write/Edit) hook for tracking hot files";
      };

      userPromptSubmit = mkOption {
        type = types.bool;
        default = true;
        description = "Enable UserPromptSubmit hook for context injection";
      };

      sessionStart = mkOption {
        type = types.bool;
        default = true;
        description = "Enable SessionStart hook for loading state";
      };

      sessionEnd = mkOption {
        type = types.bool;
        default = true;
        description = "Enable SessionEnd hook for saving state";
      };
    };

    settings = {
      maxReadChars = mkOption {
        type = types.int;
        default = 20000;
        description = "File size threshold (chars) for triggering compression";
      };

      summaryBudget = mkOption {
        type = types.int;
        default = 900;
        description = "Target token budget for compressed output";
      };

      cacheDir = mkOption {
        type = types.str;
        default = "${config.xdg.cacheHome}/scopepack";
        description = "Directory for SQLite cache and models";
      };

      embedder = mkOption {
        type = types.enum [
          "bge-small-en-v1.5"
          "titan-embed-text-v2:0"
          "cohere-embed-v4:0"
        ];
        default = "bge-small-en-v1.5";
        description = "Embedding model to use";
      };
    };
  };

  config = mkIf cfg.enable {
    # Add scopepack to PATH
    home.packages = [ cfg.package ];

    # Environment variables
    home.sessionVariables = {
      SCOPE_DAEMON_URL = "http://${cfg.daemon.host}:${toString cfg.daemon.port}";
      SCOPE_CACHE_DIR = cfg.settings.cacheDir;
      SCOPE_MAX_READ_CHARS = toString cfg.settings.maxReadChars;
      SCOPE_SUMMARY_BUDGET = toString cfg.settings.summaryBudget;
      SCOPE_EMBEDDER = cfg.settings.embedder;
    };

    # Ensure cache directory exists
    home.activation.scopepackCacheDir = lib.hm.dag.entryAfter [ "writeBoundary" ] ''
      mkdir -p "${cfg.settings.cacheDir}"
    '';

    # Install hooks to ~/.claude/hooks/
    home.file = mkIf cfg.hooks.enable (
      lib.filterAttrs (n: v: v != null) {
        ".claude/hooks/scope_pretool_read.py" =
          if cfg.hooks.preToolRead then
            {
              source = "${hooksDir}/scope_pretool_read.py";
              executable = true;
            }
          else
            null;

        ".claude/hooks/scope_posttool_write_edit.py" =
          if cfg.hooks.postToolWriteEdit then
            {
              source = "${hooksDir}/scope_posttool_write_edit.py";
              executable = true;
            }
          else
            null;

        ".claude/hooks/scope_user_prompt_submit.py" =
          if cfg.hooks.userPromptSubmit then
            {
              source = "${hooksDir}/scope_user_prompt_submit.py";
              executable = true;
            }
          else
            null;

        ".claude/hooks/scope_session_start.sh" =
          if cfg.hooks.sessionStart then
            {
              source = "${hooksDir}/scope_session_start.sh";
              executable = true;
            }
          else
            null;

        ".claude/hooks/scope_session_end.py" =
          if cfg.hooks.sessionEnd then
            {
              source = "${hooksDir}/scope_session_end.py";
              executable = true;
            }
          else
            null;
      }
    );

    # Daemon service - macOS (launchd)
    launchd.agents.scopepack-daemon = mkIf (cfg.daemon.enable && pkgs.stdenv.isDarwin) {
      enable = true;
      config = {
        Label = "com.scopepack.daemon";
        ProgramArguments = [
          "${cfg.package}/bin/scopepack-daemon"
          "--host"
          cfg.daemon.host
          "--port"
          (toString cfg.daemon.port)
        ];
        RunAtLoad = true;
        KeepAlive = true;
        StandardOutPath = "${cfg.settings.cacheDir}/daemon.log";
        StandardErrorPath = "${cfg.settings.cacheDir}/daemon.err";
        EnvironmentVariables = {
          SCOPE_CACHE_DIR = cfg.settings.cacheDir;
          SCOPE_EMBEDDER = cfg.settings.embedder;
        };
      };
    };

    # Daemon service - Linux (systemd)
    systemd.user.services.scopepack-daemon = mkIf (cfg.daemon.enable && pkgs.stdenv.isLinux) {
      Unit = {
        Description = "ScopePack compression daemon";
        After = [ "network.target" ];
      };

      Service = {
        Type = "simple";
        ExecStart = "${cfg.package}/bin/scopepack-daemon --host ${cfg.daemon.host} --port ${toString cfg.daemon.port}";
        Restart = "on-failure";
        RestartSec = 5;
        Environment = [
          "SCOPE_CACHE_DIR=${cfg.settings.cacheDir}"
          "SCOPE_EMBEDDER=${cfg.settings.embedder}"
        ];
      };

      Install = {
        WantedBy = [ "default.target" ];
      };
    };
  };
}
