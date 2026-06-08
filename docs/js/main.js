(function () {
  const nav = document.querySelector(".site-nav-inner");
  if (!nav) return;

  const links = nav.querySelectorAll('a[href^="#"]');
  const sections = [];

  links.forEach(function (link) {
    const id = link.getAttribute("href").slice(1);
    const el = document.getElementById(id);
    if (el) sections.push({ link: link, el: el });
  });

  function onScroll() {
    let current = sections[0];
    const y = window.scrollY + 100;
    sections.forEach(function (s) {
      if (s.el.offsetTop <= y) current = s;
    });
    links.forEach(function (l) {
      l.style.color = "";
      l.style.fontWeight = "";
    });
    if (current) {
      current.link.style.color = "#2563eb";
      current.link.style.fontWeight = "600";
    }
  }

  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();
})();
