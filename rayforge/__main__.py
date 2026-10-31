"""Allow running the application via ``python -m rayforge``."""

import sys

if len(sys.argv) > 1 and sys.argv[1] == "serve":
    from .serve import main

    sys.exit(main(sys.argv[2:]))

from .app import main

if __name__ == "__main__":
    main()
