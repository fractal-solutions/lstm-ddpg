// Runs the same training/backtest pipeline across several random seeds and
// prints a comparison table. Each seed gets its own `seed<SEED>_*` checkpoints.

import { join } from 'node:path';

const ROOT = import.meta.dir;

function parseArgs(argv) {
    const options = {};
    for (let i = 0; i < argv.length; i++) {
        const arg = argv[i];
        if (!arg.startsWith('--')) continue;
        const key = arg.slice(2);
        if (key === 'no-pretrain' || key === 'no-bc') {
            options[key] = true;
            continue;
        }
        const value = argv[i + 1];
        if (value === undefined || value.startsWith('--')) {
            options[key] = true;
            continue;
        }
        options[key] = value;
        i++;
    }
    return options;
}

function run(command) {
    const result = Bun.spawnSync(command, {
        cwd: ROOT,
        stdout: 'pipe',
        stderr: 'pipe'
    });
    if (result.exitCode !== 0) {
        console.error(`Command failed: ${command.join(' ')}`);
        console.error(result.stderr.toString());
        process.exit(1);
    }
    return result.stdout.toString();
}

function extract(output, label) {
    for (const line of output.split('\n')) {
        if (line.includes(label)) return line.split(/\s{2,}/).pop().trim();
    }
    return '-';
}

const args = parseArgs(process.argv.slice(2));
const seeds = String(args.seeds ?? '1,2,3').split(',').map(Number).filter(Number.isFinite);
const folds = Number(args.folds ?? 3);

const trainArgs = ['bun', 'run', 'main.js'];
const sharedArgs = [];
for (const key of ['epochs', 'steps', 'batch', 'pretrain-epochs', 'pretrain-steps', 'bc-epochs', 'bc-steps'])  {
    if (args[key] !== undefined) trainArgs.push(`--${key}`, String(args[key]));
}
for (const key of ['tf', 'bars', 'range-window', 'features', 'data', 'teacher', 'teacher-sign', 'train-fraction', 'test-end', 'embargo', 'bc-anchor']) {
    if (args[key] !== undefined) sharedArgs.push(`--${key}`, String(args[key]));
}
if (args['no-pretrain']) trainArgs.push('--no-pretrain');
if (args['no-bc']) trainArgs.push('--no-bc');

console.log(`Running ${seeds.length} seeds${args['no-pretrain'] ? ' (no pretrain)' : ''}\n`);
console.log('seed   trades  dir(L/S)   win%    PF      expR     ret%     maxDD%   folds+');

const rows = [];
const tfTag = String(args.tf ?? 'D1').toLowerCase();
for (const seed of seeds) {
    const prefix = `${tfTag}_seed${seed}`;
    run([...trainArgs, ...sharedArgs, '--seed', String(seed), '--prefix', prefix]);

    const modelSuffix = Number(args.epochs) === 0 ? 'bc_model' : 'best_model';
    const output = run(['bun', 'run', 'backtest.js', '--model', `${prefix}_${modelSuffix}`, '--no-csv', '--folds', String(folds), ...sharedArgs]);

    const trades = extract(output, 'Trades');
    const direction = (trades.match(/\(long (\d+) \/ short (\d+)\)/) ?? []).slice(1).join('/');
    const win = extract(output, 'Win rate');
    const pf = extract(output, 'Profit factor');
    const expR = extract(output, 'Expectancy');
    const ret = extract(output, 'Total return');
    const dd = extract(output, 'Max drawdown');

    const foldMatches = output.split('\n').filter(line => /^\d+\s+\d+\s+\d+\s/.test(line));
    const positiveFolds = foldMatches.filter(line => {
        const parts = line.trim().split(/\s+/);
        return Number(parts[6]) > 0;
    }).length;

    const row = {
        seed,
        trades: trades.split(' ')[0],
        direction,
        win,
        pf,
        expR,
        ret,
        dd,
        folds: `${positiveFolds}/${foldMatches.length}`
    };
    rows.push(row);

    console.log(
        `${String(seed).padEnd(6)} ${String(row.trades).padEnd(7)} ${String(row.direction).padEnd(10)} ` +
        `${String(row.win).padEnd(7)} ${String(row.pf).padEnd(7)} ${String(row.expR).padEnd(8)} ` +
        `${String(row.ret).padEnd(8)} ${String(row.dd).padEnd(8)} ${row.folds}`
    );
}

const numeric = (value) => Number(String(value).replace('%', '')) || 0;
const avgReturn = rows.reduce((sum, row) => sum + numeric(row.ret), 0) / rows.length;
const avgPF = rows.reduce((sum, row) => sum + numeric(row.pf), 0) / rows.length;
const positiveSeeds = rows.filter(row => numeric(row.ret) > 0).length;

console.log(`\nAggregate: ${positiveSeeds}/${rows.length} seeds profitable, avg return ${avgReturn.toFixed(2)}%, avg PF ${avgPF.toFixed(3)}`);
