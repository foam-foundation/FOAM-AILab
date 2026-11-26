# Tips

Short, practical tips for working on FOAM-AILab, collected as the lessons go.
Each lesson links here instead of repeating them.

## Building

### The usual build

```bash
./build.sh -Jdemo -jc
```

- `-Jdemo` adds the journals in `deployment/demo`, such as the demo user.
- `-j` deletes the runtime journals, so you start from clean data (see below).
- `-c` cleans the generated code before building.

Leave out `-j` when you want to keep what you entered in the app.

### `-j` asks before it deletes

`-j` deletes everything entered while the app ran: the runtime journals, and
also stored files and documents. So the build asks first:

- `y`: delete
- `n`: keep the journals and continue the build
- `b`: back up, then delete. `b:name` backs up to a directory with that name.

From lesson 4 on, your dataset lives in the runtime journals. Answer `b` if
there's any chance you'll want it back.

### Avoid `-y`

`-y` answers the `-j` question for you: yes, delete, with no backup. Many FOAM
projects use `-jc -y` out of habit. Avoid it once your data matters.

### A build that behaves strangely

If a build still shows old classes after `-c`, or fails in a way that makes no
sense, remove all build output and rebuild:

```bash
./build.sh -XcleanAll -o
./build.sh -Jdemo -jc
```

`-XcleanAll` removes `build/` and the app's `lib/`. `-o` only builds, without
starting the server.

### Turning a feature on for one build

`--flags:` sets a flag for one build, such as `./build.sh -Jdemo -jc --flags:ml`.
To turn a feature on for good, set it in the root `pom.js` with `setFlags`
instead (lesson 0). Write flag lists without spaces, or the build fails with
`foam: malformed flags`.

## Markdown flows

A Markdown flow is a `.md` file that starts with a `<flow .../>` tag. The build
turns it into a page in the app (lesson 0).

- **Change the text without a rebuild.** In development mode, edit the `.md` file and reload the browser.
- **Changing the `<flow>` tag needs a rebuild**, for example a new `spid` or `accessLevel`.
- **Set `spid="aILab"`** in the tag. Without it, users other than admin see an empty page (see the [SPID guide](https://github.com/foam-foundation/foam3/blob/development/doc/guides/SPID.md)).
- **Write each list item on one line.** FOAM's Markdown view doesn't yet continue a list item onto the next line.

## Server

- **Port 8080 is in use.** Another copy of the app (or something else) is running. Stop it before you build again.
- **The build fails on missing foam3 files.** The submodule isn't checked out: run `git submodule update --init`.
- **`Permission denied (publickey)` when fetching foam3.** The repository uses SSH for `foam3`. Set up SSH keys with GitHub: see the [GitHub SSH guide](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent).
