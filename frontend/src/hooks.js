import { useEffect, useState } from 'react';
import { waitForModel } from './api';

/* True once the backend model is loaded and ready to serve requests. */
export function useModelReady() {
  const [state, setState] = useState({ ready: false, error: null });

  useEffect(() => {
    let cancelled = false;
    waitForModel()
      .then(() => {
        if (!cancelled) setState({ ready: true, error: null });
      })
      .catch((e) => {
        if (!cancelled) setState({ ready: false, error: e.message });
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return state;
}
