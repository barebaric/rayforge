import React from 'react';

export default function BlogLayout({
  toc,
  sidebar,
  children,
  wrapperClassName,
}) {
  return (
    <div
      className={`container rfBlogContainer${wrapperClassName ? ` ${wrapperClassName}` : ''}`}
    >
      <div className="rfBlogGrid">
        <main className="rfBlogMain">{children}</main>
        {toc || sidebar ? (
          <aside className="rfBlogAside">
            {toc ? (
              <section className="rfBlogAsideSection">
                <div className="rfBlogAsideTitle">On this page</div>
                {toc}
              </section>
            ) : null}
            {sidebar ? (
              <section className="rfBlogAsideSection">
                <div className="rfBlogAsideTitle">Browse</div>
                {sidebar}
              </section>
            ) : null}
          </aside>
        ) : null}
      </div>
    </div>
  );
}
