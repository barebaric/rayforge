import React from 'react';
import Layout from '@theme/Layout';
import BlogSidebarContent from '@theme/BlogSidebar/Content';
import {
  BlogSidebarItemList,
  useVisibleBlogSidebarItems,
} from '@docusaurus/plugin-content-blog/client';

const ListComponent = ({items}) => {
  return <BlogSidebarItemList items={items} ulClassName="clean-list" />;
};

export default function BlogLayout(props) {
  const {toc, sidebar, children, wrapperClassName, ...layoutProps} = props;
  const sidebarItems = useVisibleBlogSidebarItems(sidebar?.items ?? []);
  const hasSidebar = Boolean(sidebar && sidebarItems.length > 0);
  const containerClassName = `container rfBlogContainer${
    wrapperClassName ? ` ${wrapperClassName}` : ''
  }`;

  return (
    <Layout {...layoutProps} wrapperClassName={wrapperClassName}>
      <div className={containerClassName}>
        <div className="rfBlogGrid">
          <main className="rfBlogMain">{children}</main>
          {toc || hasSidebar ? (
            <aside className="rfBlogAside">
              {toc ? (
                <section className="rfBlogAsideSection">
                  <div className="rfBlogAsideTitle">On this page</div>
                  {toc}
                </section>
              ) : null}
              {hasSidebar ? (
                <section className="rfBlogAsideSection">
                  <div className="rfBlogAsideTitle">{sidebar.title}</div>
                  <BlogSidebarContent
                    items={sidebarItems}
                    ListComponent={ListComponent}
                  />
                </section>
              ) : null}
            </aside>
          ) : null}
        </div>
      </div>
    </Layout>
  );
}
