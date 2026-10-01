"""Internal fresh-process worker; not an independent user-facing study entry point."""
import sys
from .common import checked
from .persistence import resume_worker

if __name__ == "__main__":
    resume_worker(checked(sys.argv[1]))
