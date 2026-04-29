# Quipper on Apple Silicon (M4) via Rosetta + GHC 8.6.5

## High-level picture

* Machine: Apple Silicon Mac (M4), macOS, user `hersh`.
* Goal: Run Quipper (0.9) reliably for testing / benchmarking.
* Constraint: Quipper works best with **GHC 8.6.x** and **x86_64**, but machine is **arm64**.
* Strategy:

  * Use **Rosetta** to run an **x86_64** Haskell toolchain.
  * Install **GHC 8.6.5** manually via `ghcup` using the official x86_64 bindist.
  * Use **Cabal** to install Quipper packages.
  * Manually wire up:

    * `quipper` and `quipper-pp` executables,
    * Quipper’s data files (`scripts/*.awk`),
    * GHC environment file (`GHC_ENVIRONMENT`).

Result: a working `quipper` command that can compile and run Quipper programs in a Rosetta shell.

---

## Directory and file layout (final state)

Key directories/files we ended up using:

* `~/.ghcup/` — ghcup-managed toolchains.

  * `~/.ghcup/ghc/8.6.5/` — GHC 8.6.5 x86_64.
  * `~/.ghcup/env` — environment file for ghcup tools.
* `~/.local/bin/` — user’s Haskell binaries (manually placed):

  * `~/.local/bin/quipper`
  * `~/.local/bin/quipper-pp`
* `~/.local/state/cabal/store/ghc-8.6.5/qppr-lngg-0.9.0.0-*/share/scripts/`
  — Quipper’s helper scripts, including `convert_template.awk`.
* `~/.cabal/share/x86_64-osx-ghc-8.6.5/quipper-language-0.9.0.0/scripts/`
  — directory we created and populated with those scripts so `quipper-pp` can find them.
* `~/.ghc/x86_64-darwin-8.6.5/environments/default`
  — **GHC environment file** created by `cabal install --lib ...` to expose Quipper libs.
* `~/.zshrc-quipper`
  — Quipper-specific shell rc file (sets PATH, ghcup env, and `GHC_ENVIRONMENT`).
* `~/quipper-src/quipper-language-0.9.0.0/`
  — local copy of the `quipper-language` source from Hackage, used to build `quipper` and `quipper-pp`.
* `~/quipper-test/`
  — directory for test programs like `And_gate.hs`.

---

## One-time setup: toolchain and packages

### 0. Prerequisites

* Rosetta 2 installed (via macOS prompt or `softwareupdate --install-rosetta`).
* Xcode Command Line Tools installed:

```bash
xcode-select --install
# In our case they were already installed.
```

### 1. Start an x86_64 (Rosetta) shell

All Haskell / Quipper work happens in a Rosetta zsh:

```bash
arch -x86_64 zsh
arch
# Should print: i386 or x86_64
```

(If it says `arm64`, you’re not in the Rosetta shell.)

### 2. Install `ghcup` (Haskell toolchain manager)

```bash
curl --proto '=https' --tlsv1.2 -sSf https://get-ghcup.haskell.org | bash

# Then in shell:
source "$HOME/.ghcup/env"
ghcup --version
```

### 3. Install GHC 8.6.5 from x86_64 bindist via ghcup

`ghcup` doesn’t have native aarch64 builds for GHC 8.6.5, so we manually use the official **x86_64** macOS tarball and tell `ghcup` to install it.

In the Rosetta shell:

```bash
cd ~/Downloads

# Download x86_64 macOS bindist for GHC 8.6.5:
curl -O https://downloads.haskell.org/~ghc/8.6.5/ghc-8.6.5-x86_64-apple-darwin.tar.xz

# Ask ghcup to “install ghc 8.6.5” from that tarball:
ghcup install ghc \
  -u "file://$PWD/ghc-8.6.5-x86_64-apple-darwin.tar.xz" \
  8.6.5

# Set it as default GHC:
ghcup set ghc 8.6.5
ghc --version
# => The Glorious Glasgow Haskell Compilation System, version 8.6.5
```

We also have Cabal installed via ghcup (either automatically, or run `ghcup install cabal latest` at some point).

### 4. Install Quipper packages with Cabal

In the Rosetta shell:

```bash
cabal update

# Try to install quipper-all (this builds almost everything)
cabal install quipper-all
```

This succeeded for:

* `quipper-language-0.9.0.0`
* `quipper-libraries-0.9.0.0`
* `quipper-tools-0.9.0.0`
* `quipper-utils`, `quipper-cabal`, `quipper-demos`, `quipper-0.9.0.0` (meta-lib)

But **failed** for:

* `quipper-algorithms-0.9.0.0` (and thus `quipper-all` as a meta-package), with:

```text
<command line>: can't load framework: Security (not found)
```

We **ignored** `quipper-algorithms` (only contains sample algorithms) and focused on core Quipper.

---

## One-time manual build of `quipper` and `quipper-pp` executables

We needed the actual `quipper` and `quipper-pp` binaries, and Cabal’s global install path was awkward, so we built them from source and copied them into `~/.local/bin`.

### 5. Get `quipper-language` source locally

```bash
mkdir -p ~/quipper-src
cd ~/quipper-src

# Download Quipper language source from Hackage
cabal get quipper-language-0.9.0.0

cd quipper-language-0.9.0.0
```

### 6. Build the `quipper` executable

```bash
cabal build exe:quipper
```

This produced (path may vary slightly):

```text
dist-newstyle/build/x86_64-osx/ghc-8.6.5/quipper-language-0.9.0.0/build/Quipper/quipper
```

We copied it into `~/.local/bin`:

```bash
mkdir -p "$HOME/.local/bin"
cp dist-newstyle/build/x86_64-osx/ghc-8.6.5/quipper-language-0.9.0.0/build/Quipper/quipper \
   "$HOME/.local/bin/quipper"
chmod +x "$HOME/.local/bin/quipper"
```

### 7. Build the `quipper-pp` executable

Similarly:

```bash
cd ~/quipper-src/quipper-language-0.9.0.0
cabal build exe:quipper-pp

# Find the built binary:
find dist-newstyle -type f -name 'quipper-pp'
# e.g. dist-newstyle/.../build/Quipper-pp/quipper-pp

cp dist-newstyle/build/x86_64-osx/ghc-8.6.5/quipper-language-0.9.0.0/build/Quipper-pp/quipper-pp \
   "$HOME/.local/bin/quipper-pp"
chmod +x "$HOME/.local/bin/quipper-pp"
```

---

## One-time fix for Quipper’s data/scripts (`convert_template.awk`)

`quipper-pp` uses an AWK script `convert_template.awk` from Quipper’s data dir. Built binary expects it under `~/.cabal/share/...`, but Cabal v2 stores data under `~/.local/state/cabal/store/...`, so we had to copy the scripts.

### 8. Locate the scripts in Cabal’s store

```bash
find "$HOME/.local/state" "$HOME/.cabal" -name 'convert_template.awk' 2>/dev/null
```

We found:

```text
/Users/hersh/.local/state/cabal/store/ghc-8.6.5/qppr-lngg-0.9.0.0-f7b17e17/share/scripts/convert_template.awk
```

So the scripts live in:

```text
SRC_SCRIPTS="/Users/hersh/.local/state/cabal/store/ghc-8.6.5/qppr-lngg-0.9.0.0-f7b17e17/share/scripts"
```

### 9. Create the path Quipper expects and copy scripts there

Quipper expects:

```text
$HOME/.cabal/share/x86_64-osx-ghc-8.6.5/quipper-language-0.9.0.0/scripts/
```

We created it and copied all scripts:

```bash
mkdir -p "$HOME/.cabal/share/x86_64-osx-ghc-8.6.5/quipper-language-0.9.0.0/scripts"

cp "$SRC_SCRIPTS"/* \
   "$HOME/.cabal/share/x86_64-osx-ghc-8.6.5/quipper-language-0.9.0.0/scripts/"
```

At this point `quipper-pp` can find `convert_template.awk`.

---

## One-time registration of Quipper libs via GHC environment

To make `import Quipper` work in plain GHC invocations (and therefore in `quipper`), we used Cabal’s `--lib` mode to create a **GHC environment file**.

### 10. Install Quipper libs with `--lib`

From any directory in the Rosetta shell:

```bash
cabal install --lib quipper-language-0.9.0.0 quipper-libraries-0.9.0.0 quipper-tools-0.9.0.0
```

This created:

```text
/Users/hersh/.ghc/x86_64-darwin-8.6.5/environments/default
```

Cabal warns that environment files can confuse tools; that’s fine here because we’re using a dedicated GHC 8.6.5 + environment specifically for Quipper.

---

## Quipper-specific shell rc file (`~/.zshrc-quipper`)

We created a custom rc file to:

* Put `~/.local/bin` on PATH (for `quipper` and `quipper-pp`).
* Source `~/.ghcup/env` (for `ghc`, `cabal`).
* Force GHC to use the Quipper environment file.

Contents (final):

```bash
# ~/.zshrc-quipper

export PATH="$HOME/.local/bin:$PATH"
[ -f "$HOME/.ghcup/env" ] && source "$HOME/.ghcup/env"

# Force GHC 8.6.5 to use the Quipper environment file
export GHC_ENVIRONMENT="$HOME/.ghc/x86_64-darwin-8.6.5/environments/default"
```

You run this only inside the **Rosetta** shell used for Quipper.

---

## How to run Quipper programs now (normal workflow)

### 1. Start a Quipper-capable shell (Rosetta + rc)

In macOS Terminal:

```bash
# Start a Rosetta zsh:
arch -x86_64 zsh --login

# Load Quipper environment:
source ~/.zshrc-quipper

# Sanity checks:
arch
# -> i386 or x86_64

ghc --version
# -> 8.6.5

which quipper
which quipper-pp
# -> /Users/hersh/.local/bin/...
```

You should also see GHC reporting that it loaded the environment file when compiling:

```text
Loaded package environment from /Users/hersh/.ghc/x86_64-darwin-8.6.5/environments/default
```

### 2. Write a Quipper program

Example: `~/quipper-test/And_gate.hs`:

```haskell
import Quipper

andGate :: Qubit -> Qubit -> Circ Qubit
andGate a b = do
  c <- qinit False
  c <- qnot c `controlled` (a, b)
  return c

main :: IO ()
main = print_simple Preview andGate
```

### 3. Compile it with `quipper`

From `~/quipper-test`:

```bash
quipper And_gate.hs
```

You should see:

```text
Loaded package environment from /Users/hersh/.ghc/x86_64-darwin-8.6.5/environments/default
[1 of 1] Compiling Main             ( And_gate.hs, And_gate.o )
Linking And_gate ...
ld: warning: ignoring duplicate libraries: '-lm'
```

This creates an executable `./And_gate` in the same directory.

### 4. Run the resulting executable

```bash
./And_gate
```

This actually runs the Quipper program and prints the circuit (ASCII or other format depending on how `main` is written and what `print_*` call you use).

---

## Notes for building a Python shim

If another user is going to design a Python shim around this, they should assume:

1. **All Quipper-related commands must run inside the Rosetta + Quipper env**
   i.e., for any subprocess call from Python, you need something equivalent to:

   * Shell: `arch -x86_64 zsh -lc 'source ~/.zshrc-quipper && quipper MyProg.hs'`

   Or:

   * Have the user manually open a Rosetta Terminal and run the Python shim from there, with `~/.zshrc-quipper` auto-sourced.

2. **Compilation pattern**:

   * Compile: `quipper MyProg.hs` → produces `./MyProg` (no automatic run).
   * Run: `./MyProg` → prints circuits / statevectors / whatever `main` does.

3. **Capturing output**:

   * The shim can:

     * `subprocess.run(["quipper", "MyProg.hs"], ...)` to compile.
     * Then `subprocess.run(["./MyProg"], capture_output=True, text=True)` to get textual output.
   * If statevector output is produced as text (e.g. via Quipper’s simulation features), the shim can parse it from stdout.

4. **Paths & binaries**:

   * `quipper` and `quipper-pp` are in `~/.local/bin`.
   * GHC 8.6.5 is in `~/.ghcup/ghc/8.6.5/`.
   * Environment file for packages is at `~/.ghc/x86_64-darwin-8.6.5/environments/default`.

5. **Packages**:

   * Core Quipper libs are available:

     * `quipper-language-0.9.0.0`
     * `quipper-libraries-0.9.0.0`
     * `quipper-tools-0.9.0.0`
   * `quipper-algorithms` and `quipper-all` remain **broken** due to macOS `Security` framework linking, and are not required for user-defined circuits.
