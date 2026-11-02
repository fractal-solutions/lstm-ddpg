// Deterministic RNG helpers. setSeed replaces Math.random for the process so
// the existing modules (weight init, replay sampling, exploration) all become
// reproducible without threading a generator through every call site.

export function setSeed(seed) {
    let a = seed >>> 0;
    Math.random = () => {
        a = (a + 0x6d2b79f5) | 0;
        let t = Math.imul(a ^ (a >>> 15), 1 | a);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

export function randn() {
    let u = 0;
    let v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

export const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
