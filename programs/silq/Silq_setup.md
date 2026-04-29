# Silq Installation Guide (macOS)

This document explains how to install the Silq compiler (`silq`) on macOS in a way that works on modern systems (e.g. macOS 15.x), and how to expose it on `PATH` so the benchmarking harness can call it.

The steps below **do not** rely on Silq’s pre-pinned LDC bundle (which segfaults on recent macOS). Instead, we install **LDC via Homebrew** and point Silq’s build script to that compiler.

> **Tested environment:** macOS, Homebrew, zsh.
> **Goal:** Obtain a working `silq` binary callable anywhere via `silq ...`.

---

## 0. Prerequisites

### 0.1. Homebrew

If you don’t have Homebrew:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Follow any on-screen instructions to add Homebrew to your `PATH`, then verify:

```bash
brew --version
```

### 0.2. Xcode Command Line Tools

You need a working C/C++ toolchain:

```bash
sudo xcode-select --install
sudo xcode-select --reset
```

---

## 1. Install LDC (D compiler) via Homebrew

Silq is written in D and needs a D compiler. Use Homebrew’s LDC instead of Silq’s bundled one:

```bash
brew install ldc
```

Confirm it’s available:

```bash
ldc2 --version
# or
ldmd2 --version  # if present
```

You should see version info printed (e.g. `LDC - the LLVM D compiler (1.41.0)`).

---

## 2. Clone the Silq repository

Pick a directory for external tools, then clone Silq:

```bash
cd ~/src   # or any other directory you like
git clone https://github.com/eth-sri/silq.git
cd silq
```

You should now see files like:

```bash
ls
# ast/  util/  library/  silq.d  build.sh  build-release.sh  dependencies-release.sh  ...
```

---

## 3. Run release dependency script (optional but recommended)

Silq ships a `dependencies-release.sh` script that downloads required D libraries and support files. It also fetches an LDC bundle you won’t ultimately use, but it’s still safe to run:

```bash
chmod +x dependencies-release.sh build-release.sh
./dependencies-release.sh
```

This may download and unpack an LDC tarball into the repo (e.g. `dmd2` / `ldc2` directories). We will **override** the compiler to use the Homebrew `ldc2` instead of this bundled one.

---

## 4. Edit `build-release.sh` to use Homebrew `ldc2`

By default, `build-release.sh` defines `LDMD` to point at the locally unpacked D compiler (from the tarball). On newer macOS versions, that toolchain may **segfault** while compiling Silq.

We fix this by pointing `LDMD` at Homebrew’s `ldc2` (or `ldmd2`) instead.

1. Open `build-release.sh` in your editor.

2. Find the line that assigns `LDMD`, which will look roughly like:

   ```sh
   LDMD=./dmd2/osx/bin/ldmd2
   # or some similar path into the downloaded toolchain
   ```

3. Replace that assignment with:

   ```sh
   LDMD="$(command -v ldmd2 || command -v ldc2)"
   ```

   So the top of the script’s build section should now effectively resolve `LDMD` to the Homebrew-installed compiler.

4. Save and close the file.

---

## 5. Build the Silq compiler

Back in the terminal, from the Silq repo:

```bash
cd ~/src/silq   # or your chosen path
./build-release.sh
```

If the override is working, this should:

* run `ldc2` on all the `.d` source files, and
* produce a `silq` binary in the repo root.

Verify:

```bash
ls -l silq
file silq
```

You want something like:

```text
-rwxr-xr-x  ... silq
silq: Mach-O 64-bit executable arm64 (or x86_64)
```

If the file exists but is not executable:

```bash
chmod +x silq
```

---

## 6. Quick sanity check (local run)

Still in the Silq repo, run the example from the official docs:

```bash
# Create a trivial Silq program
echo 'def main(){ x:=H(false); return measure(x); }' > correct.slq

# Type-check only (no output if correct)
./silq correct.slq

# Run the program (simulate)
./silq correct.slq --run
# Expected: prints 0 or 1
```

If this works, you have a functioning `silq` compiler in this directory.

---

## 7. Add `silq` to your PATH (optional but recommended)

To be able to run `silq` from anywhere (and so other tools/harnesses can call it without a hard-coded path):

```bash
cd ~/src/silq   # ensure this is the repo root where ./silq lives

# Create the bin directory if it doesn't exist
sudo mkdir -p /usr/local/bin

# Symlink the Silq binary into /usr/local/bin
sudo ln -sf "$(pwd)/silq" /usr/local/bin/silq
```

Open a **new** terminal and verify:

```bash
which silq
silq --help   # or just `silq` to see "no input files" error
```

You should see:

```text
/usr/local/bin/silq
```

and Silq responding rather than `command not found`.

If `which silq` still returns nothing, ensure `/usr/local/bin` is on your `PATH` in `~/.zshrc`:

```bash
echo 'export PATH="/usr/local/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
which silq
```
