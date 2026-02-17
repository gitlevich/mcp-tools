#!/usr/bin/env node
/**
 * AST-aware TypeScript rename via ts-morph.
 *
 * Usage: node ts_rename.js <file_path> <old_name> <new_name> <tsconfig_path>
 *
 * Outputs JSON: { "files_changed": ["relative/path.ts", ...] }
 * On error:     { "error": "message" }
 */

const { Project } = require("ts-morph");
const path = require("path");

function main() {
  const [filePath, oldName, newName, tsconfigPath] = process.argv.slice(2);

  if (!filePath || !oldName || !newName || !tsconfigPath) {
    console.log(
      JSON.stringify({
        error: "Usage: ts_rename.js <file_path> <old_name> <new_name> <tsconfig_path>",
      })
    );
    process.exit(1);
  }

  try {
    const project = new Project({ tsConfigFilePath: tsconfigPath });
    const sourceFile = project.getSourceFileOrThrow(path.resolve(filePath));

    // Find the first declaration matching oldName
    const node = findDeclaration(sourceFile, oldName);
    if (!node) {
      console.log(
        JSON.stringify({ error: `Symbol '${oldName}' not found in ${filePath}` })
      );
      process.exit(1);
    }

    // Rename across the whole project
    node.rename(newName);

    // Collect changed files
    const changed = project.getSourceFiles()
      .filter((sf) => !sf.isSaved())
      .map((sf) => sf.getFilePath());

    // Save all changes to disk
    project.saveSync();

    // Output relative paths from tsconfig dir
    const base = path.dirname(path.resolve(tsconfigPath));
    const relative = changed.map((f) => path.relative(base, f));

    console.log(JSON.stringify({ files_changed: relative }));
  } catch (err) {
    console.log(JSON.stringify({ error: err.message }));
    process.exit(1);
  }
}

/**
 * Walk top-level declarations in source file looking for a named symbol.
 * Returns the identifier node for renaming.
 */
function findDeclaration(sourceFile, name) {
  // Functions
  for (const fn of sourceFile.getFunctions()) {
    if (fn.getName() === name) return fn;
  }
  // Classes
  for (const cls of sourceFile.getClasses()) {
    if (cls.getName() === name) return cls;
  }
  // Interfaces
  for (const iface of sourceFile.getInterfaces()) {
    if (iface.getName() === name) return iface;
  }
  // Type aliases
  for (const alias of sourceFile.getTypeAliases()) {
    if (alias.getName() === name) return alias;
  }
  // Variable declarations
  for (const stmt of sourceFile.getVariableStatements()) {
    for (const decl of stmt.getDeclarations()) {
      if (decl.getName() === name) return decl;
    }
  }
  // Enums
  for (const en of sourceFile.getEnums()) {
    if (en.getName() === name) return en;
  }
  // Exported declarations (re-check via symbol)
  const sym = sourceFile.getLocal(name);
  if (sym) {
    const decls = sym.getDeclarations();
    if (decls.length > 0) return decls[0];
  }
  return null;
}

main();
