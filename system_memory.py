"""How much memory the machine has free, and how much this process uses,
asked of the operating system itself.

For the checks made before building a dense matrix of several GiB (a
16384 x 16384 float64 matrix is 2 GiB): such an allocation can succeed on
paper and then drive the machine into swap, or fail half-way through a
load, so the app refuses up front with a clear message instead. No
third-party package is needed -- psutil is not a dependency -- so this
reads the OS directly: GlobalMemoryStatusEx / GetProcessMemoryInfo on
Windows, /proc on Linux, sysconf elsewhere.

Every function returns None when the OS cannot be asked; callers must
then go ahead rather than refuse on a guess.
"""

import os
import sys

GIB = 2**30


def available_bytes():
    """Physical memory the OS could give this process now without
    swapping (Windows ullAvailPhys, Linux MemAvailable), or None."""
    try:
        if sys.platform == "win32":
            return _windows_available()
        if os.path.exists("/proc/meminfo"):
            return _proc_meminfo("MemAvailable")
        pages = os.sysconf("SC_AVPHYS_PAGES")
        return int(pages) * int(os.sysconf("SC_PAGE_SIZE"))
    except Exception:  # noqa: BLE001 -- unknown, never a crash
        return None


def total_bytes():
    """Installed physical memory, or None."""
    try:
        if sys.platform == "win32":
            return _windows_status().ullTotalPhys
        if os.path.exists("/proc/meminfo"):
            return _proc_meminfo("MemTotal")
        return int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except Exception:  # noqa: BLE001
        return None


def process_rss_bytes():
    """This process's resident memory (Windows working set), or None."""
    try:
        if sys.platform == "win32":
            return _windows_process_counters().WorkingSetSize
        with open("/proc/self/statm") as fh:
            resident_pages = int(fh.read().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE")
    except Exception:  # noqa: BLE001
        return None


def process_peak_rss_bytes():
    """This process's peak resident memory so far, or None."""
    try:
        if sys.platform == "win32":
            return _windows_process_counters().PeakWorkingSetSize
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak if sys.platform == "darwin" else peak * 1024
    except Exception:  # noqa: BLE001
        return None


def format_gib(n):
    return f"{n / GIB:.1f} GiB"


def shortfall_message(what, needed, available=None):
    """A user-facing refusal when `needed` bytes are more than the OS has
    available, or None when it is enough (or cannot be told)."""
    if available is None:
        available = available_bytes()
    if available is None or needed <= available:
        return None
    return (f"{what} needs about {format_gib(needed)} of memory, but only "
            f"{format_gib(available)} is available on this computer now. "
            f"Close other programs and try again.")


# --------------------------------------------------------------- platforms


def _proc_meminfo(key):
    with open("/proc/meminfo") as fh:
        for line in fh:
            name, _, rest = line.partition(":")
            if name == key:
                return int(rest.split()[0]) * 1024     # reported in kB
    return None


def _windows_status():
    import ctypes
    from ctypes import wintypes

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [("dwLength", wintypes.DWORD),
                    ("dwMemoryLoad", wintypes.DWORD),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("GlobalMemoryStatusEx failed")
    return status


def _windows_available():
    return int(_windows_status().ullAvailPhys)


def _windows_process_counters():
    import ctypes
    from ctypes import wintypes

    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t)]

    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
    kernel32 = ctypes.windll.kernel32
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    get_info = kernel32.K32GetProcessMemoryInfo
    # The pseudo-handle is (HANDLE)-1; without argtypes ctypes would pass
    # it as a C int and overflow.
    get_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
                         wintypes.DWORD]
    get_info.restype = wintypes.BOOL
    process = kernel32.GetCurrentProcess()
    if not get_info(process, ctypes.byref(counters), counters.cb):
        raise OSError("GetProcessMemoryInfo failed")
    return counters
