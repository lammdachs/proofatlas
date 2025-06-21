# Navigator Module Documentation

The `navigator` module provides an interactive terminal-based visualization for exploring theorem proving proofs step by step.

## Overview

The proof navigator allows users to step through a proof interactively, viewing the state at each step with a clean terminal UI using box-drawing characters.

## Features

- **Two-column layout**: Clear separation of PROCESSED and UNPROCESSED clauses
- **Given clause highlighting**: Selected clause marked with arrow (→)
- **Rule application display**: Shows which rules were applied
- **Keyboard navigation**: Simple controls for stepping through proof
- **Clean terminal UI**: Box-drawing characters for professional appearance
- **Metadata display**: Shows additional information for each step

## Usage

### Command Line

```bash
# Navigate a proof file
python -m proofatlas.navigator proof.json

# Navigate with a specific problem file
python -m proofatlas.navigator proof.json --problem problem.json
```

### Programmatic Usage

```python
from proofatlas.navigator import ProofNavigator
from proofatlas.core import load_proof

# Load a proof
proof = load_proof("proof.json")

# Create and run navigator
navigator = ProofNavigator(proof)
navigator.run()
```

## Keyboard Controls

- `n` or `→` - Next step
- `p` or `←` - Previous step  
- `q` - Quit
- `h` - Show help

## Display Format

The navigator shows each proof step with:

```
╔══════════════════════════════════════════════════════════════════════════════╗
║ PROOF NAVIGATOR - Step 2/3                                                   ║
╠══════════════════════════════╦═══════════════════════════════════════════════╣
║ PROCESSED                     ║ UNPROCESSED                                   ║
╠══════════════════════════════╬═══════════════════════════════════════════════╣
║ [0] P(a)                      ║ [0] ~P(a)  →                                  ║
║                               ║                                               ║
╚══════════════════════════════╩═══════════════════════════════════════════════╝

Given clause: ~P(a)

Applied Rules:
  resolution (parents: [0, given]) → []

[n]ext [p]rev [q]uit [h]elp
```

## Implementation Details

### Terminal Handling

The navigator uses ANSI escape codes for:
- Screen clearing: `\033[2J\033[H`
- Cursor positioning
- Clean display updates

### Raw Terminal Mode

For single-key input without pressing Enter:
```python
import termios, tty

# Save terminal settings
old_settings = termios.tcgetattr(sys.stdin)
try:
    # Enable raw mode
    tty.setraw(sys.stdin.fileno())
    # Read single character
    key = sys.stdin.read(1)
finally:
    # Restore settings
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
```

### Box Drawing

Uses Unicode box-drawing characters:
- `╔ ╗ ╚ ╝` - Corners
- `═ ║` - Horizontal and vertical lines
- `╠ ╣ ╦ ╩ ╬` - Intersections

### Clause Formatting

Clauses are formatted with:
- Index in square brackets: `[0]`
- Given clause marked with arrow: `→`
- Proper alignment and padding

## Extending the Navigator

To add new features:

1. **New display elements**: Modify `display_step()` method
2. **New keyboard commands**: Add to `handle_key()` method
3. **Additional metadata**: Update `format_metadata()` method

Example:
```python
def handle_key(self, key):
    """Handle keyboard input."""
    if key == 's':  # New command: save current step
        self.save_current_step()
    # ... existing handlers
```

## Integration with Proof Format

The navigator expects proofs with:
- List of `ProofStep` objects
- Each step containing:
  - `state`: ProofState with processed/unprocessed clauses
  - `selected_clause`: Index of given clause (optional)
  - `applied_rules`: List of RuleApplication objects
  - `metadata`: Additional information

## Error Handling

The navigator handles:
- Missing proof files
- Invalid proof format
- Terminal resize events
- Keyboard interrupts (Ctrl+C)

## Future Enhancements

Potential improvements:
- Search within proof
- Jump to specific step
- Export current view
- Syntax highlighting for clauses
- Mouse support for clicking
- Side-by-side proof comparison