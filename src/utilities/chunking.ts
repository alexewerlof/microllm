import { isStr } from 'jty'

export interface DocumentChunkMetadata {
    title: string
    level: number
    id: number
    parent?: number
    children?: number[]
    [key: string]: any
}

export interface DocumentChunk {
    content: string
    metadata: DocumentChunkMetadata
}

interface Section {
    title: string
    level: number
    id: number
    children: number[]
    parent?: number
    contentLines: string[]
}

const HEADER_REGEX = /^(#{1,6})\s*(.*?)$/

/**
 * Parses a single line to detect a markdown header.
 *
 * @example
 * ```ts
 * parseHeaderLine('## My Section') // { level: 2, title: 'My Section' }
 * parseHeaderLine('just text')     // null
 * ```
 *
 * @param line a single line of text
 * @returns the header level and title, or null if not a header
 */
function parseHeaderLine(line: string): { level: number; title: string } | null {
    const match = line.match(HEADER_REGEX)
    if (!match) return null

    return { level: match[1].length, title: (match[2] ?? '').trim() }
}

/**
 * Splits markdown lines into an ordered list of sections, each with a title,
 * level, id, and collected content lines.
 *
 * @example
 * ```ts
 * splitIntoSections(['# Title', 'Body text.', '## Sub', 'More text.'])
 * // [
 * //   { title: 'Title', level: 1, id: 1, children: [], contentLines: ['Body text.'] },
 * //   { title: 'Sub', level: 2, id: 2, children: [], contentLines: ['More text.'] },
 * // ]
 * ```
 *
 * @param lines the lines of a markdown document
 * @returns the parsed sections in document order
 */
function splitIntoSections(lines: string[]): Section[] {
    const sections: Section[] = []
    let nextId = 1

    for (const line of lines) {
        const header = parseHeaderLine(line)

        if (header) {
            sections.push({
                title: header.title,
                level: header.level,
                id: nextId++,
                children: [],
                contentLines: [],
            })
        } else if (sections.length > 0) {
            sections[sections.length - 1].contentLines.push(line)
        }
    }

    return sections
}

/**
 * Builds the `children` arrays for a flat list of sections based on header depth.
 *
 * A section is a direct child of the nearest preceding section with a smaller
 * header level (e.g. `##` is a child of the preceding `#`).
 *
 * @example
 * ```ts
 * const sections = [
 *   { id: 1, level: 1, children: [] },
 *   { id: 2, level: 2, children: [] },
 *   { id: 3, level: 2, children: [] },
 * ]
 * assignChildren(sections)
 * // sections[0].children is now [2, 3]
 * ```
 *
 * @param sections the ordered list of sections with id and level
 */
function assignChildren(sections: { id: number; level: number; children: number[] }[]): void {
    for (let i = 0; i < sections.length; i++) {
        for (let j = i + 1; j < sections.length; j++) {
            if (sections[j].level <= sections[i].level) break

            if (sections[j].level === sections[i].level + 1) {
                sections[i].children.push(sections[j].id)
            }
        }
    }
}

/**
 * Produces a markdown header line from a level and title.
 *
 * @example
 * ```ts
 * toHeaderLine(2, 'Setup') // '## Setup'
 * ```
 *
 * @param level the header depth (1–6)
 * @param title the header text
 * @returns the markdown header line
 */
function toHeaderLine(level: number, title: string): string {
    return '#'.repeat(level) + ' ' + title
}

/**
 * Assigns `parent` ids and builds a breadcrumb string for each section in a
 * single pass using an ancestor stack.
 *
 * The breadcrumb is a series of markdown header lines representing the full
 * path from the document root to the section itself.
 *
 * @example
 * ```ts
 * const sections = splitIntoSections(['# Root', 'R.', '## Child', 'C.'])
 * assignBreadcrumbs(sections)
 * // sections[1].parent === 1
 * // returned map: { 1 => '# Root', 2 => '# Root\n## Child' }
 * ```
 *
 * @param sections the ordered list of sections (mutates `parent` field)
 * @returns a map from section id to its breadcrumb header string
 */
function assignBreadcrumbs(sections: Section[]): Map<number, string> {
    const breadcrumbs = new Map<number, string>()
    const stack: Section[] = []

    for (const section of sections) {
        while (stack.length > 0 && stack[stack.length - 1].level >= section.level) {
            stack.pop()
        }

        if (stack.length > 0) {
            section.parent = stack[stack.length - 1].id
        }

        const headerLines = [
            ...stack.map((s) => toHeaderLine(s.level, s.title)),
            toHeaderLine(section.level, section.title),
        ]
        breadcrumbs.set(section.id, headerLines.join('\n'))

        stack.push(section)
    }

    return breadcrumbs
}

/**
 * Converts a parsed section into a {@link DocumentChunk}, merging in optional
 * global metadata and trimming content whitespace.
 *
 * @example
 * ```ts
 * const section = { title: 'Intro', level: 1, id: 1, children: [2], contentLines: ['Hello.'] }
 * sectionToChunk(section)
 * // { content: 'Hello.', metadata: { title: 'Intro', level: 1, id: 1, children: [2] } }
 * ```
 *
 * @param section a parsed section
 * @returns the document chunk
 */
function sectionToChunk(section: Section): DocumentChunk {
    return {
        content: section.contentLines.join('\n').trim(),
        metadata: {
            title: section.title,
            level: section.level,
            id: section.id,
            ...(section.parent != null ? { parent: section.parent } : {}),
            children: section.children,
        },
    }
}

/**
 * Chunks a markdown document into flat pieces based on headers, with each
 * chunk's content prefixed by its full hierarchy of markdown headers.
 *
 * For example, a `### Detail` section under `## Concept` under `# Guide`
 * will have its content prefixed with `# Guide\n## Concept\n### Detail\n`.
 *
 * @example
 * ```ts
 * headerChunk('# Guide\nIntro.\n## Setup\nInstall npm.', { docId: 1 })
 * // [
 * //   { content: '# Guide\nIntro.', metadata: { docId: 1, title: 'Guide', level: 1, id: 1, children: [2] } },
 * //   { content: '# Guide\n## Setup\nInstall npm.', metadata: { docId: 1, title: 'Setup', level: 2, id: 2, parent: 1, children: [] } },
 * // ]
 * ```
 *
 * The `children` and `parent` fields in the metadata allow you to understand
 * the hierarchy of the chunks.
 *
 * @param markdown the markdown document to chunk
 */
export function headerChunk(markdown: string): DocumentChunk[] {
    if (!isStr(markdown)) {
        throw new TypeError('Expected markdown to be a string, got ' + typeof markdown)
    }

    const sections = splitIntoSections(markdown.split('\n'))
    assignChildren(sections)
    const breadcrumbs = assignBreadcrumbs(sections)

    return sections
        .filter((s) => s.contentLines.join('\n').trim().length > 0)
        .map((s) => {
            const chunk = sectionToChunk(s)
            return { ...chunk, content: breadcrumbs.get(s.id)! + '\n' + chunk.content }
        })
}

export const _test = {
    parseHeaderLine,
    splitIntoSections,
    assignChildren,
    toHeaderLine,
    assignBreadcrumbs,
    sectionToChunk,
}
