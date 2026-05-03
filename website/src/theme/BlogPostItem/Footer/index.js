import React from 'react';
import { useBlogPost } from '@docusaurus/plugin-content-blog/client';
import TagsListInline from '@theme/TagsListInline';
import BlogPostItemFooterOriginal from '@theme-original/BlogPostItem/Footer';

function BlogMetaBar() {
  const { metadata } = useBlogPost();
  const readMinutes =
    typeof metadata.readingTime === 'number'
      ? Math.max(1, Math.ceil(metadata.readingTime))
      : null;

  return (
    <div className="rfBlogMetaBar">
      <div className="rfBlogMetaLeft">
        {metadata.tags?.length ? <TagsListInline tags={metadata.tags} /> : null}
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
  return (
    <>
      <BlogPostItemFooterOriginal {...props} />
      <BlogMetaBar />
    </>
  );
}

