import React from 'react';
import Link from '@docusaurus/Link';
import { translate } from '@docusaurus/Translate';
import { useBlogPost } from '@docusaurus/plugin-content-blog/client';

function BlogTags({ tags }) {
  const sortedTags = [...tags].sort((left, right) =>
    left.label.localeCompare(right.label, undefined, { sensitivity: 'base' }),
  );

  return (
    <>
      {sortedTags.map((tag) => (
        <Link key={tag.permalink} to={tag.permalink} className="tag">
          {tag.label}
        </Link>
      ))}
    </>
  );
}

function BlogMetaBar() {
  const { metadata } = useBlogPost();
  const readMinutes =
    typeof metadata.readingTime === 'number'
      ? Math.max(1, Math.ceil(metadata.readingTime))
      : null;

  return (
    <div className="rfBlogMetaBar">
      <div className="rfBlogMetaLeft">
        {metadata.tags?.length ? <BlogTags tags={metadata.tags} /> : null}
      </div>
      <div className="rfBlogMetaRight">
        <span>{metadata.formattedDate}</span>
        {readMinutes !== null ? <span>·</span> : null}
        {readMinutes !== null ? <span>{readMinutes} min read</span> : null}
      </div>
    </div>
  );
}

export default function BlogPostItemFooter(props) {
  const { metadata } = useBlogPost();

  return (
    <footer {...props}>
      {metadata.truncated ? (
        <div className="margin-top--sm">
          <Link
            to={metadata.permalink}
            className="button button--secondary button--sm"
            aria-label={translate(
              {
                id: 'theme.blog.post.readMoreLabel',
                message: 'Read more about {title}',
                description:
                  'The ARIA label for the link to full blog posts from excerpts',
              },
              { title: metadata.title },
            )}
          >
            {translate({
              id: 'theme.blog.post.readMore',
              message: 'Read more',
              description:
                'The label used in blog post item excerpts to link to full blog posts',
            })}
          </Link>
        </div>
      ) : null}
      <BlogMetaBar />
    </footer>
  );
}
