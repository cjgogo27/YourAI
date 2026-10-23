import { spawnSync } from 'node:child_process';
import { copyFileSync, existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

// Git handles authentication itself. This script never reads access tokens.
const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
function command(executable, args, cwd = root, capture = false, allowFailure = false) {
  const result = spawnSync(executable, args, {
    cwd, encoding: 'utf8', stdio: capture ? 'pipe' : 'inherit', windowsHide: true,
  });
  if (result.error) throw result.error;
  if (result.status !== 0 && !allowFailure) throw new Error(`${executable} exited with ${result.status}`);
  return capture ? (result.stdout || '').trim() : result.status;
}
const git = (args, cwd = root, capture = false, allowFailure = false) => command('git', args, cwd, capture, allowFailure);
function copyBuild(source, destination) {
  mkdirSync(destination, { recursive: true });
  for (const entry of readdirSync(source, { withFileTypes: true })) {
    const sourcePath = join(source, entry.name);
    const destinationPath = join(destination, entry.name);
    if (entry.isDirectory()) copyBuild(sourcePath, destinationPath);
    else if (entry.isFile()) copyFileSync(sourcePath, destinationPath);
    else throw new Error('Build output must contain only regular files and directories.');
  }
}
const remote = git(['remote', 'get-url', 'origin'], root, true);
const sourceCommit = git(['rev-parse', '--short', 'HEAD'], root, true);
const authorName = git(['config', 'user.name'], root, true, true) || git(['log', '-1', '--format=%an'], root, true);
const authorEmail = git(['config', 'user.email'], root, true, true) || git(['log', '-1', '--format=%ae'], root, true);

command(process.execPath, [join(root, 'node_modules', 'vite', 'bin', 'vite.js'), 'build']);
const dist = join(root, 'dist');
if (!existsSync(join(dist, 'index.html'))) throw new Error('No built index.html found.');
const checkout = mkdtempSync(join(root, '.pages-deploy-'));
try {
  git(['init', '-b', 'gh-pages'], checkout);
  git(['remote', 'add', 'origin', remote], checkout);
  const branchExists = git(['ls-remote', '--exit-code', '--heads', 'origin', 'gh-pages'], checkout, false, true);
  if (branchExists === 0) {
    git(['fetch', '--depth=1', 'origin', 'gh-pages'], checkout);
    git(['checkout', '-B', 'gh-pages', 'FETCH_HEAD'], checkout);
  } else if (branchExists !== 2) {
    throw new Error('Unable to inspect the publishing branch; no files were published.');
  }
  const cnamePath = join(checkout, 'CNAME');
  const cname = existsSync(cnamePath) ? readFileSync(cnamePath) : null;
  // Only remove tracked build files inside this newly created disposable clone.
  if (dirname(resolve(checkout)) !== root || !checkout.startsWith(join(root, '.pages-deploy-'))) {
    throw new Error('Unexpected publication directory.');
  }
  git(['rm', '-r', '--ignore-unmatch', '.'], checkout);
  copyBuild(dist, checkout);
  writeFileSync(join(checkout, '.nojekyll'), '');
  if (cname && !existsSync(join(dist, 'CNAME'))) writeFileSync(cnamePath, cname);
  git(['add', '--all'], checkout);
  if (git(['diff', '--cached', '--quiet'], checkout, false, true) === 0) {
    console.log('Published files already match this build.');
  } else {
    git(['-c', `user.name=${authorName}`, '-c', `user.email=${authorEmail}`,
      'commit', '-m', `Publish YourAI Lab from ${sourceCommit}`], checkout);
    // No force push: concurrent updates must be resolved before publishing.
    git(['push', 'origin', 'HEAD:gh-pages'], checkout);
    console.log('Build published to gh-pages. GitHub Pages will deploy this commit.');
  }
} finally {
  const checkedPath = resolve(checkout);
  if (dirname(checkedPath) === root && checkedPath.startsWith(join(root, '.pages-deploy-'))) {
    rmSync(checkedPath, { recursive: true, force: true });
  }
}
