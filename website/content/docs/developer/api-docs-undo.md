# Undo Manager API Documentation

The Undo/Redo framework provides a transactional history manager based on
the Command pattern. It supports explicit transactions for multi-part actions
and automatic coalescing for rapid, identical actions.

---

## Class: HistoryManager

`rayforge.core.undo.history.HistoryManager` manages undo/redo history
for document operations.

### Constructor

```python
HistoryManager()
```

Initializes a new, empty history manager.

### Properties

- **`undo_stack`** (`List[Command]`): Stack of commands that can be undone.
- **`redo_stack`** (`List[Command]`): Stack of commands that can be redone.
- **`changed`** (`Signal`): Signal that emits when history changes.
- **`in_transaction`** (`bool`): Whether a transaction is currently active.
- **`transaction_commands`** (`List[Command]`): Commands in current transaction.
- **`transaction_name`** (`str`): Name of current transaction.

### Command Execution

- **`execute(command: Command)`**
  Executes a command and adds it to the history, possibly coalescing it
  with the previous command.

- **`add(command: Command)`**
  Adds a command that has already been executed to the history, possibly
  coalescing it with the previous command.

### Undo/Redo Operations

- **`undo()`**
  Undoes the last action from the undo stack and moves it to the redo
  stack.

- **`redo()`**
  Redoes the last undone action from the redo stack and moves it to the
  undo stack.

- **`undo_to(target_command: Command)`**
  Undoes all actions up to and including the specified target command.

- **`redo_to(target_command: Command)`**
  Redoes all actions up to and including the specified target command.

- **`can_undo() -> bool`**
  Returns `True` if there are actions available to undo.

- **`can_redo() -> bool`**
  Returns `True` if there are actions available to redo.

### Transactions

- **`transaction(name: str = "Transaction") -> Iterator[_TransactionContextProxy]`**
  Context manager for grouping commands into a single transaction. If the
  transaction completes successfully, commands are grouped into a single history
  entry. If only one command is executed, it is "unwrapped" and added
  directly. Otherwise, commands are bundled into a `CompositeCommand`. If an
  exception occurs, all commands executed within the transaction are undone.

  ```python
  with history_manager.transaction("My Changes") as t:
      t.execute(SetterCommand(...))
      t.set_label("A better name")  # Optional
  ```

- **`begin_transaction(name: str = "Transaction")`**
  Starts an explicit transaction. All subsequent commands executed will be
  grouped together until `end_transaction()` is called. Raises `RuntimeError`
  if a transaction is already active.

- **`end_transaction()`**
  Ends the current transaction, creates a `CompositeCommand`, and adds it
  to the history. If only one command is in the transaction, it is
  unwrapped and added directly.

- **`abort_transaction()`**
  Aborts the current transaction, discarding any commands that were added
  since it began. Note: This does not undo the commands themselves; that
  is handled by the transaction context manager's exception block.

### Utility Methods

- **`clear()`**
  Clears all undo and redo history, including any active transaction.

---

## Class: Command

`rayforge.core.undo.command.Command` is the abstract base class for all
undoable actions.

### Constructor

```python
Command(name: Optional[str] = None, on_change_callback: Optional[Callable[[], None]] = None)
```

### Properties

- **`name`** (`Optional[str]`): Display name for the command (e.g., for UI).
- **`on_change_callback`** (`Optional[Callable[[], None]]`): Optional callback
  executed after command execution or undo.
- **`timestamp`** (`float`): Unix timestamp when the command was created.

### Abstract Methods

- **`execute() -> None`**
  Performs the action. Must be implemented by subclasses.

- **`undo() -> None`**
  Reverts the action. Must be implemented by subclasses.

### Coalescing Methods

- **`can_coalesce_with(next_command: Command) -> bool`**
  Checks if the next command can be merged into this one without modifying
  the state of either command. Returns `False` by default.

- **`coalesce_with(next_command: Command) -> bool`**
  Attempts to merge the next command into this one. If successful, this
  command's state is updated with the newer command's state and it returns
  `True`. Otherwise, it returns `False`. Returns `False` by default.

---

## Built-in Command Classes

### CompositeCommand

`rayforge.core.undo.composite_cmd.CompositeCommand` groups several commands
into a single transaction.

**Constructor:**

```python
CompositeCommand(commands: List[Command], name: str, on_change_callback: Optional[Callable[[], None]] = None)
```

**Methods:**

- **`execute()`**: Executes all child commands in order.
- **`undo()`**: Undoes all child commands in reverse order.
- **`can_coalesce_with()`**: Checks if another CompositeCommand can be merged
  (must have the same number of child commands and each child must be
  coalescable).
- **`coalesce_with()`**: Merges another CompositeCommand if all their
  respective child commands can be coalesced.

### ChangePropertyCommand

`rayforge.core.undo.property_cmd.ChangePropertyCommand` changes a single
property on an object.

**Constructor:**

```python
ChangePropertyCommand(
    target: Any,
    property_name: str,
    new_value: Any,
    old_value: Any = _sentinel,
    setter_method_name: Optional[str] = None,
    on_change_callback: Optional[Callable[[], None]] = None,
    name: Optional[str] = None,
)
```

**Parameters:**

- `target`: The object whose property will be changed.
- `property_name`: Name of the property to change.
- `new_value`: The new value to set.
- `old_value`: The old value (fetched from the object if not provided).
- `setter_method_name`: Optional name of a setter method to use instead of
  direct attribute assignment.

**Methods:**

- **`execute()`**: Sets the property to the new value.
- **`undo()`**: Restores the old value.
- **`can_coalesce_with()`**: Checks if another command affects the same
  property on the same object.
- **`coalesce_with()`**: Merges by updating to the newer command's value.

### ListItemCommand

`rayforge.core.undo.list_cmd.ListItemCommand` adds or removes an item
from a list-like container.

**Constructor:**

```python
ListItemCommand(
    owner_obj: Any,
    item: Any,
    undo_command: str,
    redo_command: str,
    on_change_callback: Optional[Callable[[], None]] = None,
    name: Optional[str] = None,
)
```

**Parameters:**

- `owner_obj`: The object containing the list.
- `item`: The item to add or remove.
- `undo_command`: Name of the method to call for undo (e.g., "remove_child").
- `redo_command`: Name of the method to call for redo (e.g., "add_child").

**Methods:**

- **`execute()`**: Executes the redo action.
- **`undo()`**: Executes the undo action.

### ReorderListCommand

`rayforge.core.undo.list_cmd.ReorderListCommand` handles the reordering of a
list.

**Constructor:**

```python
ReorderListCommand(
    target_obj: Any,
    list_property_name: str,
    new_list: List[Any],
    setter_method_name: Optional[str] = None,
    on_change_callback: Optional[Callable[[], None]] = None,
    name: Optional[str] = None,
)
```

**Parameters:**

- `target_obj`: The object containing the list.
- `list_property_name`: Name of the list property to reorder.
- `new_list`: The new order for the list.
- `setter_method_name`: Optional name of a setter method to use instead of
  direct attribute assignment.

**Methods:**

- **`execute()`**: Applies the new order to the list.
- **`undo()`**: Restores the original order of the list.

### DictItemCommand

`rayforge.core.undo.dict_cmd.DictItemCommand` changes a value for a
specific key in a dictionary.

**Constructor:**

```python
DictItemCommand(
    target_dict: Dict[str, Any],
    key: str,
    new_value: Any,
    name: str,
    on_change_callback: Optional[Callable[[], Any]] = None,
)
```

**Parameters:**

- `target_dict`: The dictionary to modify.
- `key`: The key whose value will be changed.
- `new_value`: The new value to set for the key.
- `name`: The user-facing name for this command.

**Methods:**

- **`execute()`**: Sets the new value in the dictionary.
- **`undo()`**: Restores the old value in the dictionary (or removes the
  key if the old value was `None`).
- **`can_coalesce_with()`**: Checks if another command affects the same
  dictionary key.
- **`coalesce_with()`**: Merges by updating to the newer command's value.

### SetterCommand

`rayforge.core.undo.setter_cmd.SetterCommand` calls a setter method with
arbitrary arguments.

**Constructor:**

```python
SetterCommand(
    target: Any,
    setter_method_name: str,
    new_args: Tuple[Any, ...],
    old_args: Tuple[Any, ...],
    on_change_callback: Optional[Callable[[], None]] = None,
    name: Optional[str] = None,
)
```

**Parameters:**

- `target`: The object containing the setter method.
- `setter_method_name`: Name of the setter method to call.
- `new_args`: The new arguments to pass to the setter.
- `old_args`: The old arguments to pass to the setter for undo.

**Methods:**

- **`execute()`**: Executes the setter with the new arguments.
- **`undo()`**: Executes the setter with the old arguments to revert.
- **`can_coalesce_with()`**: Checks if another command affects the same
  object and setter method.
- **`coalesce_with()`**: Merges by updating to the newer command's
  arguments.

---

## Constants

- **`COALESCE_THRESHOLD`** (`float = 0.5`): Maximum time in seconds between
  two commands to be considered for automatic coalescing.

---

## Usage Examples

### Basic Command Execution

```python
from rayforge.core.undo import HistoryManager, ChangePropertyCommand

history = HistoryManager()

# Execute a command
cmd = ChangePropertyCommand(obj, "name", "New Name")
history.execute(cmd)

# Undo
history.undo()

# Redo
history.redo()
```

### Using Transactions

```python
# Group multiple commands into a single undoable action
with history.transaction("Move Items") as t:
    t.execute(ChangePropertyCommand(item1, "x", 10.0))
    t.execute(ChangePropertyCommand(item1, "y", 20.0))
    t.execute(ChangePropertyCommand(item2, "x", 15.0))

# All three changes will be undone/redone together
history.undo()
```

### Custom Command

```python
from rayforge.core.undo import Command

class MyCustomCommand(Command):
    def __init__(self, target, new_value, **kwargs):
        super().__init__(**kwargs)
        self.target = target
        self.new_value = new_value
        self.old_value = target.some_value

    def execute(self):
        self.target.some_value = self.new_value

    def undo(self):
        self.target.some_value = self.old_value

    def can_coalesce_with(self, next_command):
        return (
            isinstance(next_command, MyCustomCommand)
            and self.target is next_command.target
        )

    def coalesce_with(self, next_command):
        if not self.can_coalesce_with(next_command):
            return False
        self.new_value = next_command.new_value
        self.timestamp = next_command.timestamp
        return True
```
