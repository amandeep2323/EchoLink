/**
 * Tokenizer client — calls the isolated avatar backend (FastAPI on
 * 127.0.0.1:8770) to turn text into an ordered Sign_Token sequence.
 */

import type { TokenizeResponse } from './types';

const TOKENIZE_URL = 'http://127.0.0.1:8770/tokenize';
const HEALTH_URL = 'http://127.0.0.1:8770/health';

export class TokenizerError extends Error {}

export async function tokenize(
  text: string,
  fingerspellUnknown = true
): Promise<TokenizeResponse> {
  let res: Response;
  try {
    res = await fetch(TOKENIZE_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, options: { fingerspellUnknown } }),
    });
  } catch {
    throw new TokenizerError('translation unavailable');
  }
  if (!res.ok) {
    throw new TokenizerError(`tokenizer error ${res.status}`);
  }
  return (await res.json()) as TokenizeResponse;
}

export async function tokenizerHealthy(): Promise<boolean> {
  try {
    const res = await fetch(HEALTH_URL, { cache: 'no-cache' });
    return res.ok;
  } catch {
    return false;
  }
}
