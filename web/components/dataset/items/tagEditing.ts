/**
 * The logic of tagging a text word by word, kept free of React and of the `@public/` alias so `bun test` can run it.
 *
 * A text's words are what Python's `str.split()` gives (`server/omnitrain/domain/token_tags.py`): runs of Unicode
 * whitespace separate them and nothing else does, so "Hanoi," is one word. The server counts the tags against that
 * definition and refuses a list that is off by one, so the editor has to split exactly the same way. JavaScript's own
 * `\s` is not the same set (it has U+FEFF and lacks U+001C-U+001F and U+0085), hence the explicit list.
 */

export const MAX_TAG_LENGTH = 64
/** The tag a word outside any entity has; a prediction falls back to it for a word the model never read. */
export const OUTSIDE = 'O'

/** What `str.isspace()` calls whitespace (all of it is in the Basic Multilingual Plane, so UTF-16 units suffice). */
const isSpace = (code: number) =>
  (code >= 0x09 && code <= 0x0d) ||
  (code >= 0x1c && code <= 0x20) ||
  code === 0x85 ||
  code === 0xa0 ||
  code === 0x1680 ||
  (code >= 0x2000 && code <= 0x200a) ||
  code === 0x2028 ||
  code === 0x2029 ||
  code === 0x202f ||
  code === 0x205f ||
  code === 0x3000

export const splitWords = (text: string): string[] => {
  const words: string[] = []
  let start = -1
  for (let i = 0; i < text.length; i++) {
    if (isSpace(text.charCodeAt(i))) {
      if (start >= 0) words.push(text.slice(start, i))
      start = -1
    } else if (start < 0) {
      start = i
    }
  }
  if (start >= 0) words.push(text.slice(start))
  return words
}

/** A tag is one non-empty "word" of at most `MAX_TAG_LENGTH` characters: a snapshot joins tags with single spaces. */
export const isValidTag = (tag: string) =>
  tag.length > 0 && [...tag].length <= MAX_TAG_LENGTH && splitWords(tag).length === 1 && splitWords(tag)[0] === tag

/**
 * The tags to show for a text of `wordCount` words, '' for a word with none yet. Stored tags that do not fit (the
 * text was edited after it was tagged) are not shown as if they did: the server refuses them for the same reason.
 */
export const tagsForWords = (stored: unknown, wordCount: number): string[] =>
  Array.isArray(stored) && stored.length === wordCount && stored.every((t) => typeof t === 'string')
    ? [...(stored as string[])]
    : new Array<string>(wordCount).fill('')

/** `tag` on the words from `from` to `to` (either order, both included). */
export const setRange = (tags: string[], from: number, to: number, tag: string): string[] => {
  const [low, high] = from <= to ? [from, to] : [to, from]
  return tags.map((t, i) => (i >= low && i <= high ? tag : t))
}

export const setTag = (tags: string[], index: number, tag: string): string[] => setRange(tags, index, index, tag)

/** Every word that has no tag gets `tag` (the usual "everything else is O"). */
export const fillUntagged = (tags: string[], tag: string): string[] => tags.map((t) => t || tag)

export const taggedCount = (tags: string[]) => tags.filter((t) => t !== '').length

/** Every word has a tag, which is what the server accepts (exactly one per word). */
export const isComplete = (tags: string[]) => tags.length > 0 && taggedCount(tags) === tags.length

export const sameTags = (a: string[], b: string[]) => a.length === b.length && a.every((t, i) => t === b[i])

/**
 * Words tagged `I-x` that do not continue a `B-x` or `I-x` right before them. Training and scoring still read such a
 * tag (an entity starts there), but it is nearly always a slip, so the editor points at it.
 */
export const strayInside = (tags: string[]): number[] =>
  tags.flatMap((tag, i) => {
    if (!tag.startsWith('I-')) return []
    const type = tag.slice(2)
    const before = tags[i - 1] ?? ''
    return before === `B-${type}` || before === `I-${type}` ? [] : [i]
  })
