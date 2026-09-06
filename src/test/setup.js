import { cleanup } from '@testing-library/react';
import '@testing-library/jest-dom/vitest';
import { afterEach } from 'vitest';

if (typeof window !== 'undefined') {
  window.scrollTo = () => {};
}

afterEach(() => {
  cleanup();
});
