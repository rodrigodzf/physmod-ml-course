const inputList = Deno.env.get("QUARTO_USE_FILE_FOR_PROJECT_INPUT_FILES");
const inputs = inputList
  ? await Deno.readTextFile(inputList)
  : Deno.env.get("QUARTO_PROJECT_INPUT_FILES") ?? "";

const rendersWeekFive = inputs
  .split(/\r?\n/)
  .some((input) => input.replaceAll("\\", "/") === "week05/slides/index.qmd");

if (!rendersWeekFive) Deno.exit(0);

const projectDir = Deno.cwd();
const widgetDir = `${projectDir}/week05/widgets/modal-training-loop`;
const outputDir = `${projectDir}/week05/slides/widgets`;
const bundleName = "modal-training-loop.html";
const npmExecutable = Deno.build.os === "windows" ? "npm.cmd" : "npm";

async function npm(...args: string[]) {
  try {
    const { success } = await new Deno.Command(npmExecutable, {
      args,
      cwd: widgetDir,
      stdout: "inherit",
      stderr: "inherit",
    }).spawn().status;
    if (!success) throw new Error(`npm ${args[0]} failed for the Week 5 diagram.`);
  } catch (error) {
    if (error instanceof Deno.errors.NotFound) {
      throw new Error("Install Node.js and npm to render the Week 5 slides.");
    }
    throw error;
  }
}

const lock = await Deno.open(`${widgetDir}/.quarto-build.lock`, {
  create: true,
  write: true,
});
try {
  await lock.lock(true);
  const lockfile = await Deno.readTextFile(`${widgetDir}/package-lock.json`);
  const installedLockfilePath = `${widgetDir}/node_modules/.quarto-package-lock.json`;
  let installedLockfile;
  try {
    await Deno.stat(`${widgetDir}/node_modules/vite/package.json`);
    installedLockfile = await Deno.readTextFile(installedLockfilePath);
  } catch (error) {
    if (!(error instanceof Deno.errors.NotFound)) throw error;
  }
  if (installedLockfile !== lockfile) {
    await npm("ci", "--include=dev", "--no-audit", "--no-fund");
    await Deno.writeTextFile(installedLockfilePath, lockfile);
  }

  await Deno.mkdir(outputDir, { recursive: true });
  const buildDir = await Deno.makeTempDir({ dir: outputDir, prefix: ".modal-training-loop-" });
  try {
    await npm("run", "build", "--", "--outDir", buildDir);
    await Deno.rename(`${buildDir}/${bundleName}`, `${outputDir}/${bundleName}`);
  } finally {
    await Deno.remove(buildDir, { recursive: true });
  }
} finally {
  lock.close();
}
