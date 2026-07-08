import { Link, Route, Routes } from "react-router-dom";
import InvoiceDetail from "./pages/InvoiceDetail.jsx";
import InvoiceList from "./pages/InvoiceList.jsx";
import Upload from "./pages/Upload.jsx";

export default function App() {
  return (
    <div className="app">
      <header className="header">
        <h1>Invoice Parser</h1>
        <nav>
          <Link to="/">Invoices</Link>
          <Link to="/upload">Upload</Link>
        </nav>
      </header>
      <main className="main">
        <Routes>
          <Route path="/" element={<InvoiceList />} />
          <Route path="/upload" element={<Upload />} />
          <Route path="/invoices/:id" element={<InvoiceDetail />} />
        </Routes>
      </main>
    </div>
  );
}
