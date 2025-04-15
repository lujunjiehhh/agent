import React, { useState, useEffect } from 'react';
import { Table, Upload, Button, Modal, message } from 'antd';
import { UploadOutlined, EyeOutlined } from '@ant-design/icons';
import type { UploadProps } from 'antd';
import Editor from '@monaco-editor/react';
import axios from 'axios';

interface Document {
  id: string;
  filename: string;
  file_type: string;
  created_at: string;
  processed_at: string;
}

const DocumentList: React.FC = () => {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [previewVisible, setPreviewVisible] = useState(false);
  const [previewContent, setPreviewContent] = useState('');
  const [selectedDoc, setSelectedDoc] = useState<Document | null>(null);

  useEffect(() => {
    fetchDocuments();
  }, []);

  const fetchDocuments = async () => {
    try {
      const response = await axios.get('http://localhost:8000/api/documents');
      setDocuments(response.data);
    } catch (error) {
      message.error('获取文档列表失败');
    }
  };

  const handlePreview = async (doc: Document) => {
    try {
      const response = await axios.get(`http://localhost:8000/api/documents/${doc.id}/preview`);
      setPreviewContent(response.data.content);
      setSelectedDoc(doc);
      setPreviewVisible(true);
    } catch (error) {
      message.error('获取文档预览失败');
    }
  };

  const uploadProps: UploadProps = {
    name: 'file',
    action: 'http://localhost:8000/api/documents/upload',
    onChange(info) {
      if (info.file.status === 'done') {
        message.success(`${info.file.name} 上传成功`);
        fetchDocuments();
      } else if (info.file.status === 'error') {
        message.error(`${info.file.name} 上传失败`);
      }
    },
  };

  const columns = [
    {
      title: '文件名',
      dataIndex: 'filename',
      key: 'filename',
    },
    {
      title: '文件类型',
      dataIndex: 'file_type',
      key: 'file_type',
    },
    {
      title: '上传时间',
      dataIndex: 'created_at',
      key: 'created_at',
    },
    {
      title: '处理时间',
      dataIndex: 'processed_at',
      key: 'processed_at',
    },
    {
      title: '操作',
      key: 'action',
      render: (_: any, record: Document) => (
        <Button
          type="link"
          icon={<EyeOutlined />}
          onClick={() => handlePreview(record)}
        >
          预览
        </Button>
      ),
    },
  ];

  return (
    <div>
      <div style={{ marginBottom: 16 }}>
        <Upload {...uploadProps}>
          <Button icon={<UploadOutlined />}>上传文档</Button>
        </Upload>
      </div>
      <Table columns={columns} dataSource={documents} rowKey="id" />
      
      <Modal
        title={`文档预览 - ${selectedDoc?.filename}`}
        open={previewVisible}
        onCancel={() => setPreviewVisible(false)}
        width={800}
        footer={null}
      >
        <Editor
          height="500px"
          defaultLanguage="text"
          value={previewContent}
          options={{
            readOnly: true,
            minimap: { enabled: false },
            scrollBeyondLastLine: false,
          }}
        />
      </Modal>
    </div>
  );
};

export default DocumentList; 