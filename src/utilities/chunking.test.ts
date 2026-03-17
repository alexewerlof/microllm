import { describe, test } from 'node:test'
import assert from 'node:assert/strict'
import { headerChunk, _test } from './chunking.js'

const { parseHeaderLine, splitIntoSections, assignChildren, toHeaderLine, assignBreadcrumbs, sectionToChunk } = _test

describe(_test.parseHeaderLine.name, () => {
    test('parses a level-1 header', () => {
        const result = parseHeaderLine('# Hello World')
        assert.deepStrictEqual(result, { level: 1, title: 'Hello World' })
    })

    test('parses headers at different levels', () => {
        assert.deepStrictEqual(parseHeaderLine('## Section'), { level: 2, title: 'Section' })
        assert.deepStrictEqual(parseHeaderLine('### Sub'), { level: 3, title: 'Sub' })
        assert.deepStrictEqual(parseHeaderLine('###### Deep'), { level: 6, title: 'Deep' })
    })

    test('parses headers without a space after the hashes', () => {
        const result = parseHeaderLine('##NoSpace')
        assert.deepStrictEqual(result, { level: 2, title: 'NoSpace' })
    })

    test('returns null for non-header lines', () => {
        assert.strictEqual(parseHeaderLine('just text'), null)
        assert.strictEqual(parseHeaderLine(''), null)
        assert.strictEqual(parseHeaderLine('  # indented'), null)
    })

    test('works fine even when the title text is missing', () => {
        assert.deepStrictEqual(parseHeaderLine('#'), { level: 1, title: '' })
        assert.deepStrictEqual(parseHeaderLine('## '), { level: 2, title: '' })
    })

    test('trims whitespace from the title', () => {
        const result = parseHeaderLine('##   Spaced Title   ')
        assert.deepStrictEqual(result, { level: 2, title: 'Spaced Title' })
    })
})

describe(splitIntoSections.name, () => {
    test('splits lines into sections by header', () => {
        const sections = splitIntoSections(['# Title', 'Body text.', '## Sub', 'More text.'])

        assert.strictEqual(sections.length, 2)
        assert.strictEqual(sections[0].title, 'Title')
        assert.strictEqual(sections[0].id, 1)
        assert.deepStrictEqual(sections[0].contentLines, ['Body text.'])
        assert.strictEqual(sections[1].title, 'Sub')
        assert.strictEqual(sections[1].id, 2)
        assert.deepStrictEqual(sections[1].contentLines, ['More text.'])
    })

    test('ignores text before the first header', () => {
        const sections = splitIntoSections(['Preamble', '# Title', 'Body.'])

        assert.strictEqual(sections.length, 1)
        assert.strictEqual(sections[0].title, 'Title')
    })

    test('returns empty array when there are no headers', () => {
        const sections = splitIntoSections(['Just text.', 'More text.'])
        assert.deepStrictEqual(sections, [])
    })

    test('assigns sequential ids starting from 1', () => {
        const sections = splitIntoSections(['# A', '## B', '### C'])

        assert.strictEqual(sections[0].id, 1)
        assert.strictEqual(sections[1].id, 2)
        assert.strictEqual(sections[2].id, 3)
    })
})

describe(_test.assignChildren.name, () => {
    test('assigns direct children based on header levels', () => {
        const sections = [
            { id: 1, level: 1, children: [] as number[] },
            { id: 2, level: 2, children: [] as number[] },
            { id: 3, level: 2, children: [] as number[] },
        ]
        assignChildren(sections)

        assert.deepStrictEqual(sections[0].children, [2, 3])
        assert.deepStrictEqual(sections[1].children, [])
        assert.deepStrictEqual(sections[2].children, [])
    })

    test('stops assigning children when a same-level sibling is reached', () => {
        const sections = [
            { id: 1, level: 1, children: [] as number[] },
            { id: 2, level: 2, children: [] as number[] },
            { id: 3, level: 1, children: [] as number[] },
            { id: 4, level: 2, children: [] as number[] },
        ]
        assignChildren(sections)

        assert.deepStrictEqual(sections[0].children, [2])
        assert.deepStrictEqual(sections[2].children, [4])
    })

    test('handles deeply nested hierarchy', () => {
        const sections = [
            { id: 1, level: 1, children: [] as number[] },
            { id: 2, level: 2, children: [] as number[] },
            { id: 3, level: 3, children: [] as number[] },
        ]
        assignChildren(sections)

        assert.deepStrictEqual(sections[0].children, [2])
        assert.deepStrictEqual(sections[1].children, [3])
        assert.deepStrictEqual(sections[2].children, [])
    })

    test('handles an empty list', () => {
        const sections: { id: number; level: number; children: number[] }[] = []
        assignChildren(sections)
        assert.deepStrictEqual(sections, [])
    })
})

describe(_test.toHeaderLine.name, () => {
    test('produces a level-1 header', () => {
        assert.strictEqual(toHeaderLine(1, 'Title'), '# Title')
    })

    test('produces a level-3 header', () => {
        assert.strictEqual(toHeaderLine(3, 'Deep'), '### Deep')
    })

    test('produces a level-6 header', () => {
        assert.strictEqual(toHeaderLine(6, 'Deepest'), '###### Deepest')
    })
})

describe(_test.assignBreadcrumbs.name, () => {
    test('assigns parent and breadcrumb for a simple hierarchy', () => {
        const sections = splitIntoSections(['# Root', 'R.', '## Child', 'C.'])
        const breadcrumbs = assignBreadcrumbs(sections)

        assert.strictEqual(sections[0].parent, undefined)
        assert.strictEqual(sections[1].parent, 1)
        assert.strictEqual(breadcrumbs.get(1), '# Root')
        assert.strictEqual(breadcrumbs.get(2), '# Root\n## Child')
    })

    test('assigns correct parents across separate subtrees', () => {
        const sections = splitIntoSections(['# A', 'a.', '## B', 'b.', '# C', 'c.', '## D', 'd.'])
        const breadcrumbs = assignBreadcrumbs(sections)

        assert.strictEqual(sections[1].parent, 1)
        assert.strictEqual(sections[3].parent, 3)
        assert.strictEqual(breadcrumbs.get(2), '# A\n## B')
        assert.strictEqual(breadcrumbs.get(4), '# C\n## D')
    })

    test('builds full breadcrumb for three levels of nesting', () => {
        const sections = splitIntoSections(['# Root', 'R.', '## Mid', 'M.', '### Leaf', 'L.'])
        const breadcrumbs = assignBreadcrumbs(sections)

        assert.strictEqual(sections[2].parent, 2)
        assert.strictEqual(breadcrumbs.get(3), '# Root\n## Mid\n### Leaf')
    })

    test('handles an empty list', () => {
        const breadcrumbs = assignBreadcrumbs([])
        assert.strictEqual(breadcrumbs.size, 0)
    })

    test('handles siblings correctly', () => {
        const sections = splitIntoSections(['# Root', 'R.', '## A', 'a.', '## B', 'b.'])
        const breadcrumbs = assignBreadcrumbs(sections)

        assert.strictEqual(sections[1].parent, 1)
        assert.strictEqual(sections[2].parent, 1)
        assert.strictEqual(breadcrumbs.get(2), '# Root\n## A')
        assert.strictEqual(breadcrumbs.get(3), '# Root\n## B')
    })
})

describe(sectionToChunk.name, () => {
    test('converts a section to a chunk with trimmed content', () => {
        const section = { title: 'Title', level: 1, id: 1, children: [2], contentLines: ['  Hello.  '] }
        const chunk = sectionToChunk(section)

        assert.strictEqual(chunk.content, 'Hello.')
        assert.deepStrictEqual(chunk.metadata, { title: 'Title', level: 1, id: 1, children: [2] })
    })

    test('includes parent when present', () => {
        const section = { title: 'Sub', level: 2, id: 2, children: [], parent: 1, contentLines: ['Text.'] }
        const chunk = sectionToChunk(section)

        assert.strictEqual(chunk.metadata.parent, 1)
    })

    test('omits parent when absent', () => {
        const section = { title: 'Root', level: 1, id: 1, children: [], contentLines: ['Text.'] }
        const chunk = sectionToChunk(section)

        assert.strictEqual('parent' in chunk.metadata, false)
    })

    test('merges globalMetadata into chunk metadata', () => {
        const section = { title: 'Title', level: 1, id: 1, children: [], contentLines: ['Body.'] }
        const chunk = sectionToChunk(section, { docId: 42, source: 'test' })

        assert.strictEqual(chunk.metadata.docId, 42)
        assert.strictEqual(chunk.metadata.source, 'test')
        assert.strictEqual(chunk.metadata.title, 'Title')
    })

    test('joins multi-line content with newlines', () => {
        const section = { title: 'T', level: 1, id: 1, children: [], contentLines: ['Line 1', 'Line 2'] }
        const chunk = sectionToChunk(section)

        assert.strictEqual(chunk.content, 'Line 1\nLine 2')
    })
})

describe(headerChunk.name, () => {
    test('chunks a simple document with breadcrumb headers', () => {
        const md = [
            '# Top title',
            'Some description.',
            '## Section 1',
            'Content for section 1.',
            '## Section 2',
            'Content for section 2.',
        ].join('\n')

        const chunks = headerChunk(md)

        assert.strictEqual(chunks.length, 3)
        assert.deepStrictEqual(chunks[0], {
            content: '# Top title\nSome description.',
            metadata: { title: 'Top title', level: 1, id: 1, children: [2, 3] },
        })
        assert.deepStrictEqual(chunks[1], {
            content: '# Top title\n## Section 1\nContent for section 1.',
            metadata: { title: 'Section 1', level: 2, id: 2, parent: 1, children: [] },
        })
        assert.deepStrictEqual(chunks[2], {
            content: '# Top title\n## Section 2\nContent for section 2.',
            metadata: { title: 'Section 2', level: 2, id: 3, parent: 1, children: [] },
        })
    })

    test('includes globalMetadata in each chunk', () => {
        const md = '# Title\nBody text.'
        const chunks = headerChunk(md, { docId: 42, source: 'test' })

        assert.strictEqual(chunks.length, 1)
        assert.strictEqual(chunks[0].metadata.docId, 42)
        assert.strictEqual(chunks[0].metadata.source, 'test')
        assert.strictEqual(chunks[0].metadata.title, 'Title')
    })

    test('returns empty array for empty string', () => {
        assert.deepStrictEqual(headerChunk(''), [])
    })

    test('returns empty array for text with no headers', () => {
        assert.deepStrictEqual(headerChunk('Just plain text without any headers.'), [])
    })

    test('filters out headers with no content underneath', () => {
        const md = '# Title\n## Empty Section\n## Has Content\nSome text.'
        const chunks = headerChunk(md)

        assert.strictEqual(chunks.length, 1)
        assert.strictEqual(chunks[0].metadata.title, 'Has Content')
    })

    test('preserves multi-line content within a section', () => {
        const md = [
            '# Title',
            'Line one.',
            'Line two.',
            '',
            'Line three after blank.',
        ].join('\n')

        const chunks = headerChunk(md)

        assert.strictEqual(chunks.length, 1)
        assert.strictEqual(chunks[0].content, '# Title\nLine one.\nLine two.\n\nLine three after blank.')
    })

    test('handles three levels of nesting', () => {
        const md = [
            '# Root',
            'Root content.',
            '## Child A',
            'Child A content.',
            '### Grandchild',
            'Grandchild content.',
            '## Child B',
            'Child B content.',
        ].join('\n')

        const chunks = headerChunk(md)

        assert.strictEqual(chunks.length, 4)
        assert.strictEqual(chunks[0].content, '# Root\nRoot content.')
        assert.strictEqual(chunks[1].content, '# Root\n## Child A\nChild A content.')
        assert.strictEqual(chunks[2].content, '# Root\n## Child A\n### Grandchild\nGrandchild content.')
        assert.strictEqual(chunks[3].content, '# Root\n## Child B\nChild B content.')

        assert.deepStrictEqual(chunks[0].metadata.children, [2, 4])
        assert.strictEqual(chunks[0].metadata.parent, undefined)
        assert.deepStrictEqual(chunks[1].metadata.children, [3])
        assert.strictEqual(chunks[1].metadata.parent, 1)
        assert.deepStrictEqual(chunks[2].metadata.children, [])
        assert.strictEqual(chunks[2].metadata.parent, 2)
        assert.deepStrictEqual(chunks[3].metadata.children, [])
        assert.strictEqual(chunks[3].metadata.parent, 1)
    })

    test('handles markdown with only headers and no body text', () => {
        const md = '# A\n## B\n### C'
        const chunks = headerChunk(md)

        assert.strictEqual(chunks.length, 0)
    })

    test('ignores text before the first header', () => {
        const md = 'Preamble text.\n# Title\nBody.'
        const chunks = headerChunk(md)

        assert.strictEqual(chunks.length, 1)
        assert.strictEqual(chunks[0].content, '# Title\nBody.')
        assert.strictEqual(chunks[0].metadata.title, 'Title')
    })

    test('throws TypeError when markdown is not a string', () => {
        assert.throws(
            () => headerChunk(123 as any),
            { name: 'TypeError', message: /Expected markdown to be a string/ },
        )
    })

    test('throws TypeError when globalMetadata is not a plain object', () => {
        assert.throws(
            () => headerChunk('# A\ntext', 'bad' as any),
            { name: 'TypeError', message: /Expected globalMetadata to be a plain object/ },
        )
    })

    test('handles real-world markdown with bullet lists', () => {
        const md = [
            '# Service Levels',
            'Overview of service levels.',
            '## SLI',
            '- Latency',
            '- Availability',
            '- Error rate',
            '## SLO',
            'Target value for an SLI.',
        ].join('\n')

        const chunks = headerChunk(md)

        assert.strictEqual(chunks.length, 3)
        assert.strictEqual(chunks[1].content, '# Service Levels\n## SLI\n- Latency\n- Availability\n- Error rate')
    })

    test('handles the user example with three levels', () => {
        const md = [
            '# Doc title',
            'Intro text',
            '## Concept 1',
            'Description for concept 1.',
            '## Concept 2',
            'Description for concept 2.',
            '### Nuance on concept 2',
            'Description for nuance on concept 2',
            '### Example on concept 2',
            'Example for concept 2',
        ].join('\n')

        const chunks = headerChunk(md)

        assert.strictEqual(chunks[0].content, '# Doc title\nIntro text')
        assert.strictEqual(chunks[1].content, '# Doc title\n## Concept 1\nDescription for concept 1.')
        assert.strictEqual(chunks[2].content, '# Doc title\n## Concept 2\nDescription for concept 2.')
        assert.strictEqual(chunks[3].content, '# Doc title\n## Concept 2\n### Nuance on concept 2\nDescription for nuance on concept 2')
        assert.strictEqual(chunks[4].content, '# Doc title\n## Concept 2\n### Example on concept 2\nExample for concept 2')
    })
})
