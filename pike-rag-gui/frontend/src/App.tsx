import React from 'react';
import { Layout, Menu } from 'antd';
import { BrowserRouter as Router, Route, Routes, Link } from 'react-router-dom';
import {
  FileOutlined,
  QuestionCircleOutlined,
  ApiOutlined,
  NodeIndexOutlined,
} from '@ant-design/icons';

import DocumentList from './components/DocumentList';
import QAPanel from './components/QAPanel';
import KnowledgeGraph from './components/KnowledgeGraph';
import ReasoningFlow from './components/ReasoningFlow';

const { Header, Sider, Content } = Layout;

const App: React.FC = () => {
  return (
    <Router>
      <Layout style={{ minHeight: '100vh' }}>
        <Header style={{ padding: 0, background: '#fff' }}>
          <h1 style={{ margin: '0 24px', lineHeight: '64px' }}>PIKE-RAG GUI</h1>
        </Header>
        <Layout>
          <Sider width={200} style={{ background: '#fff' }}>
            <Menu
              mode="inline"
              defaultSelectedKeys={['1']}
              style={{ height: '100%', borderRight: 0 }}
            >
              <Menu.Item key="1" icon={<FileOutlined />}>
                <Link to="/documents">文档管理</Link>
              </Menu.Item>
              <Menu.Item key="2" icon={<QuestionCircleOutlined />}>
                <Link to="/qa">问答系统</Link>
              </Menu.Item>
              <Menu.Item key="3" icon={<NodeIndexOutlined />}>
                <Link to="/knowledge-graph">知识图谱</Link>
              </Menu.Item>
              <Menu.Item key="4" icon={<ApiOutlined />}>
                <Link to="/reasoning">推理过程</Link>
              </Menu.Item>
            </Menu>
          </Sider>
          <Layout style={{ padding: '24px' }}>
            <Content style={{ background: '#fff', padding: 24, margin: 0, minHeight: 280 }}>
              <Routes>
                <Route path="/documents" element={<DocumentList />} />
                <Route path="/qa" element={<QAPanel />} />
                <Route path="/knowledge-graph" element={<KnowledgeGraph />} />
                <Route path="/reasoning" element={<ReasoningFlow />} />
              </Routes>
            </Content>
          </Layout>
        </Layout>
      </Layout>
    </Router>
  );
};

export default App; 