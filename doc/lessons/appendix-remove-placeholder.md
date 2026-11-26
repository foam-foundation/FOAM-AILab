# Appendix: Removing the placeholder model

The FOAM generator gives every new app a placeholder model named after the
app: here, `AILab`, with its own **AILab** menu. FOAM-AILab doesn't use it,
but it does no harm, so the lessons leave it in place.

This appendix is optional. Removing the model is a good exercise, though: it
shows every place a model is wired into a FOAM app. When you add a model of
your own, you touch the same places.

> [!IMPORTANT]
> The lesson tags (`lesson-00`, `lesson-01`, ...) keep the placeholder. If you
> remove it in your own branch and later merge a lesson tag into that branch,
> expect merge conflicts in `journals/menus.jrl` and
> `journals/groupPermissionJunctions.jrl`. Keep your version of those lines.

## The naming pattern

The generator derives every name from the model's name. Knowing the pattern
helps you find each place:

| What | Name |
|---|---|
| Model | `com.foamdev.ailab.AILab` |
| DAO service | `aILabDAO` |
| Permission to use the service | `service.aILabDAO` |
| Permissions on the records | `aILab.create`, `aILab.read.*`, `aILab.update.*`, `aILab.delete.*` |
| Menu | `aILab.aILab` |
| Permission to see the menu | `menu.read.aILab.aILab` |

Don't confuse these with names that belong to the app, not the model: the
`aILab` group, the `aILab` service provider (`spid`), and `serviceprovider.read.aILab`.
Those stay.

## The seven places

### 1. The model: `src/com/foamdev/ailab/AILab.js`

The class itself: an `id`, a `name` and a `description`. Delete the file.

### 2. The POM: `src/com/foamdev/ailab/pom.js`

The POM lists the model and its test folder. Remove both entries, but keep the
POM: the folder also holds `About.md`, and the build finds it through this POM.

```js
foam.POM({
  name: 'aILab'
});
```

### 3. The DAO service: `journals/services.jrl`

The `aILabDAO` service stores the records. It's the only entry in the file, so
the file can be deleted, or left with no entries for the models you'll add.

> [!WARNING]
> The generated service has `.setAuthorize(false)`. With it, the DAO doesn't
> check the record permissions (`aILab.read.*` and the others): anyone who may
> use the service can read and change every record. Don't copy this into a
> real model. The Dataset's DAO, for example, uses `.setAuthorize(true)`.

### 4. The menu: `journals/menus.jrl`

Remove the `aILab.aILab` menu entry, the one with `"label":"AILab"` and a
`DAOMenu2` handler. Keep `aILab.about`.

### 5. The permissions: `journals/groupPermissionJunctions.jrl`

Remove the six lines under `// specific model` that name the model:

```js
p({class:"foam.core.auth.GroupPermissionJunction",sourceId:"aILab",targetId:"service.aILabDAO"})
p({class:"foam.core.auth.GroupPermissionJunction",sourceId:"aILab",targetId:"aILab.create"})
p({class:"foam.core.auth.GroupPermissionJunction",sourceId:"aILab",targetId:"aILab.read.*"})
p({class:"foam.core.auth.GroupPermissionJunction",sourceId:"aILab",targetId:"aILab.update.*"})
p({class:"foam.core.auth.GroupPermissionJunction",sourceId:"aILab",targetId:"aILab.delete.*"})
p({class:"foam.core.auth.GroupPermissionJunction",sourceId:"aILab",targetId:"menu.read.aILab.aILab"})
```

Keep `menu.read.aILab.about` and everything above `// specific model`.

### 6. The theme: `journals/themes.jrl`

The generator made the AILab menu the page the app opens on. Lesson 0 already
changed `defaultMenu` to `["aILab.about"]`, so there's nothing left to do here.
Without lesson 0, the app would open on a menu that no longer exists.

### 7. The test: `src/com/foamdev/ailab/test/`

The folder holds `AILabTest.js` (it saves an `AILab` record and checks that it
gets an id), its `pom.js`, and `tests.jrl`, which registers the test. Delete
the whole folder; step 2 already removed it from the POM.

## Rebuild and check

Removed classes need a clean build, and `-j` drops the records you may have
saved in the AILab screen:

```bash
./build.sh -Jdemo -jc
```

- [ ] The build finishes without errors.
- [ ] Admin's menu has no **AILab** entry; the app still opens on **About**.
- [ ] Demo sees only **About**.

If the build complains about a class that no longer exists, see the
[Tips](tips.md#a-build-that-behaves-strangely) page.

Back to [Lesson 0](00-setup.md).
