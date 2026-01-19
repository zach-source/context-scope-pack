{
  pkgs,
  lib,
  config,
  inputs,
  ...
}:

{
  # https://devenv.sh/basics/
  env.SCOPE_DAEMON_URL = "http://127.0.0.1:18765";
  env.SCOPE_CACHE_DIR = "${config.env.DEVENV_ROOT}/.cache/scopepack";
  env.TRANSFORMERS_CACHE = "${config.env.DEVENV_ROOT}/.cache/huggingface";
  env.PYTHONPATH = "${config.env.DEVENV_ROOT}/src:$PYTHONPATH";

  # https://devenv.sh/packages/
  packages = with pkgs; [
    git
    sqlite
    # For potential Rust extensions
    # rustc
    # cargo
  ];

  # https://devenv.sh/languages/
  languages.python = {
    enable = true;
    version = "3.11";
    venv = {
      enable = true;
      requirements = ''
        # Core dependencies
        sentence-transformers>=2.2.0
        transformers>=4.35.0
        torch>=2.0.0

        # HTTP server
        fastapi>=0.104.0
        uvicorn[standard]>=0.24.0
        httpx>=0.25.0

        # Database
        aiosqlite>=0.19.0

        # Utilities
        pydantic>=2.5.0
        numpy>=1.24.0

        # MCP Server
        mcp[cli]>=1.0.0

        # Development
        pytest>=7.4.0
        pytest-asyncio>=0.21.0
        ruff>=0.1.0
        mypy>=1.7.0

        # AWS Bedrock support
        boto3>=1.34.0
      '';
    };
  };

  # https://devenv.sh/processes/
  processes = {
    scope-daemon.exec = "python -m scopepack.daemon";
  };

  # https://devenv.sh/services/
  # No external services needed - SQLite is file-based

  # https://devenv.sh/scripts/
  scripts = {
    # Start the daemon in foreground for debugging
    daemon-dev.exec = ''
      cd $DEVENV_ROOT
      python -m scopepack.daemon --reload
    '';

    # Start the MCP server
    mcp-server.exec = ''
      cd $DEVENV_ROOT
      python -m scopepack.mcp_server
    '';

    # Run tests
    test.exec = ''
      cd $DEVENV_ROOT
      pytest tests/ -v
    '';

    # Format and lint
    lint.exec = ''
      cd $DEVENV_ROOT
      ruff check src/ tests/ --fix
      ruff format src/ tests/
    '';

    # Type check
    typecheck.exec = ''
      cd $DEVENV_ROOT
      mypy src/scopepack/
    '';

    # Initialize the database
    db-init.exec = ''
      cd $DEVENV_ROOT
      python -c "from scopepack.db import init_db; import asyncio; asyncio.run(init_db())"
    '';

    # Download models (first-time setup)
    models-download.exec = ''
            cd $DEVENV_ROOT
            python -c "
      from sentence_transformers import SentenceTransformer
      from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

      print('Downloading embedding model...')
      SentenceTransformer('BAAI/bge-small-en-v1.5')

      print('Downloading summarization model...')
      AutoModelForSeq2SeqLM.from_pretrained('sshleifer/distilbart-cnn-12-6')
      AutoTokenizer.from_pretrained('sshleifer/distilbart-cnn-12-6')

      print('Models downloaded successfully!')
      "
    '';
  };

  enterShell = ''
    echo "ScopePack development environment"
    echo ""
    echo "Available commands:"
    echo "  daemon-dev     - Start daemon in dev mode with auto-reload"
    echo "  mcp-server     - Start MCP server (stdio transport)"
    echo "  test           - Run tests"
    echo "  lint           - Format and lint code"
    echo "  typecheck      - Run mypy type checking"
    echo "  db-init        - Initialize SQLite database"
    echo "  models-download - Download ML models (first-time setup)"
    echo ""
    echo "Environment variables:"
    echo "  SCOPE_DAEMON_URL=$SCOPE_DAEMON_URL"
    echo "  SCOPE_CACHE_DIR=$SCOPE_CACHE_DIR"
    echo ""

    # Ensure cache directory exists
    mkdir -p "$SCOPE_CACHE_DIR"
  '';

  # https://devenv.sh/pre-commit-hooks/
  git-hooks.hooks = {
    ruff.enable = true;
    # mypy.enable = true;  # Enable once types are stable
  };

  # https://devenv.sh/tasks/
  # tasks = { };

  # See full reference at https://devenv.sh/reference/options/
}
