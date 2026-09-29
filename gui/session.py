## Shared application state

# External Imports
from PySide6.QtCore import QObject, Signal


class Session(QObject):
    '''
    The propeller being worked on, shared by every tab, and whether a long-running (JAX) task is active.
    Only one task runs at a time so the solvers never compete for the CPU.
    '''
    propeller_changed = Signal(object)      # new Propeller (design or configuration changed)
    dirty_changed = Signal(bool)            # unsaved changes
    busy_changed = Signal(bool, str)        # (busy, description of the running task)

    def __init__(self, prop, path=None, parent=None):
        super().__init__(parent)
        self.prop = prop
        self.path = path            # file the propeller was loaded from / saved to
        self.dirty = False
        self.busy = False
        self.task_name = ""

    @property
    def name(self) -> str:
        return self.prop.name or "untitled"

    def set_propeller(self, prop, dirty=True, notify=True, path=None):
        '''
        Replaces the current propeller. notify=False skips propeller_changed, for updates that leave the
        design untouched (e.g. a solved induced velocity kept as the warm start).
        '''
        self.prop = prop
        if path is not None:
            self.path = path
        if notify:
            self.propeller_changed.emit(prop)
        self.set_dirty(dirty)

    def set_dirty(self, dirty: bool):
        if dirty != self.dirty:
            self.dirty = dirty
            self.dirty_changed.emit(dirty)

    def try_begin(self, task_name: str) -> bool:
        '''
        Claims the task slot; False if another task is already running.
        '''
        if self.busy:
            return False
        self.busy, self.task_name = True, task_name
        self.busy_changed.emit(True, task_name)
        return True

    def end(self):
        self.busy, self.task_name = False, ""
        self.busy_changed.emit(False, "")
