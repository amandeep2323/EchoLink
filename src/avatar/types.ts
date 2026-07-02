/** Shared AvatarLink types — mirrors the tokenizer response schema. */

export interface WordToken {
  type: 'word';
  gloss: string;
  clipId: string;
  source: 'model4';
}

export interface FingerspellLetter {
  char: string;
  poseId: string;
}

export interface FingerspellToken {
  type: 'fingerspell';
  word: string;
  letters: FingerspellLetter[];
}

export type SignToken = WordToken | FingerspellToken;

export interface TokenizeResponse {
  input: string;
  tokens: SignToken[];
}
