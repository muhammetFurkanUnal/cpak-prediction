// CPAK angle computation and classification
// Angles match Python backend: always returns the acute angle (≤ 90°)

function _vecAngle(v1x, v1y, v2x, v2y) {
  const dot = v1x * v2x + v1y * v2y;
  const m1  = Math.hypot(v1x, v1y);
  const m2  = Math.hypot(v2x, v2y);
  if (!m1 || !m2) return 0;
  const val = Math.max(-1, Math.min(1, dot / (m1 * m2)));
  const ang = Math.acos(val) * 180 / Math.PI;
  return ang <= 90 ? ang : 180 - ang;
}

// dots: { key: DotHandle | {ix, iy} }
// Returns { ldfa, mpta } or null if required dots are missing
function computeAngles(dots) {
  const need = ['femurHead','femurNotch','femurLateral','femurMedial',
                'finalAnkleMiddle','tibiaIntercondiler','tibiaLateral','tibiaMedial'];
  if (need.some(k => !dots[k])) return null;

  const d = key => ({ ix: dots[key].ix, iy: dots[key].iy });

  const fh  = d('femurHead');   const fn  = d('femurNotch');
  const fl  = d('femurLateral');const fm  = d('femurMedial');
  const am  = d('finalAnkleMiddle'); const ti = d('tibiaIntercondiler');
  const tl  = d('tibiaLateral'); const tm  = d('tibiaMedial');

  const ldfa = _vecAngle(fn.ix - fh.ix, fn.iy - fh.iy,  fm.ix - fl.ix, fm.iy - fl.iy);
  const mpta = _vecAngle(ti.ix - am.ix, ti.iy - am.iy,  tm.ix - tl.ix, tm.iy - tl.iy);

  return { ldfa, mpta };
}

// Returns full CPAK classification object
function classifyCPAK(ldfa, mpta) {
  const ahka = mpta - ldfa;
  const jlo  = mpta + ldfa;

  const ahkaCat = ahka < -2 ? 'Varus' : ahka > 2 ? 'Valgus' : 'Neutral';
  const jloCat  = jlo > 183 ? 'Apex Proximal' : jlo < 177 ? 'Apex Distal' : 'Neutral';

  const typeMap = {
    'Varus:Apex Proximal':   'I',
    'Varus:Neutral':         'II',
    'Varus:Apex Distal':     'III',
    'Neutral:Apex Proximal': 'IV',
    'Neutral:Neutral':       'V',
    'Neutral:Apex Distal':   'VI',
    'Valgus:Apex Proximal':  'VII',
    'Valgus:Neutral':        'VIII',
    'Valgus:Apex Distal':    'IX',
  };

  return {
    ldfa, mpta, ahka, jlo,
    ahkaCat, jloCat,
    cpakType: typeMap[`${ahkaCat}:${jloCat}`] || '?',
  };
}

// All 9 types ordered for the matrix display (row = aHKA, col = JLO)
const CPAK_MATRIX = [
  ['I',   'II',   'III'],
  ['IV',  'V',    'VI'],
  ['VII', 'VIII', 'IX'],
];
const CPAK_ROW_LABELS = ['Varus', 'Neutral', 'Valgus'];
const CPAK_COL_LABELS = ['Apex Proximal', 'Neutral', 'Apex Distal'];
