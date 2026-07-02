import 'react';

// Electron frameless-window drag regions use the non-standard
// `-webkit-app-region` CSS property. Augment React's CSSProperties so
// `WebkitAppRegion` type-checks in inline styles.
declare module 'react' {
  interface CSSProperties {
    WebkitAppRegion?: 'drag' | 'no-drag';
  }
}
